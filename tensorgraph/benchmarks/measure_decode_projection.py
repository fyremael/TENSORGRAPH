"""Whole-subgraph and projection-only timing with allocation traces.

Device samples amortize one 1000-invocation captured batch. Host samples time
one invocation followed by stream synchronization. Neither is token throughput.
"""
import random
import time


def measure_projection(torch, state, x, residual, qweight, out, cell):
    normalized = state['outputs'][1].clone()
    projected = torch.empty_like(state['outputs'][2])
    def library_projection():
        torch.mm(normalized, qweight, out=projected)
    def projection_function(n, w, y):
        torch.mm(n, w, out=y)
    compiled_projection = torch.compile(projection_function, fullgraph=True)
    def inductor_projection():
        compiled_projection(normalized, qweight, projected)
    jobs = {**{'whole_'+n:fn for n,fn in state['lanes'].items()},
            'projection_library':library_projection,
            'projection_inductor':inductor_projection}
    expected = (normalized.float() @ qweight.float()).to(projected.dtype)
    for fn in (library_projection, inductor_projection):
        fn()
        atol, rtol = ((0.02, 0.02) if projected.dtype == torch.float16 else (0.0625, 0.04))
        torch.testing.assert_close(projected, expected, atol=atol, rtol=rtol)
    prefix = f"{cell['dtype']}-{cell['batch']}-{cell['hidden']}"
    record = {'warmup':20, 'rounds':5, 'host_samples_per_round':1000,
              'device_invocations_per_batch':1000, 'seed':731,
              'scope':'resident weights/inputs; persistent caller outputs; no input update copies',
              'projection_control':'same rounded normalized input, independent persistent QKV output',
              'allocation':{}, 'capture':{}, 'timings':[]}
    single = {}
    batches = {}
    def profile(fn, key):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as prof:
            for _ in range(3): fn()
            torch.cuda.synchronize()
        trace = f'alloc-{prefix}-{key}.json'
        prof.export_chrome_trace(str(out/trace))
        record['allocation'][key] = {'profiled_calls':3,'base_bytes':base,
            'peak_extra_bytes':torch.cuda.max_memory_allocated()-base,'trace':trace}
    def capture(fn, count):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        t = time.perf_counter_ns()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(count): fn()
        torch.cuda.synchronize()
        return graph, time.perf_counter_ns()-t
    for name, fn in jobs.items():
        for _ in range(20): fn()
        torch.cuda.synchronize()
        profile(fn, name+'-ordinary')
        single[name], one_ns = capture(fn, 1)
        profile(single[name].replay, name+'-replay')
        batches[name], batch_ns = capture(fn, 1000)
        for _ in range(3): batches[name].replay()
        torch.cuda.synchronize()
        record['capture'][name] = {'single_capture_ns':one_ns,'batch_capture_ns':batch_ns,
            'reserved_bytes':torch.cuda.memory_reserved()}
    rng = random.Random(731)
    for round_id in range(5):
        order = list(jobs)
        rng.shuffle(order)
        for name in order:
            for mode in ('ordinary_host','graph_host'):
                fn = jobs[name] if mode=='ordinary_host' else single[name].replay
                samples = []
                for _ in range(1000):
                    torch.cuda.synchronize()
                    t = time.perf_counter_ns()
                    fn()
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter_ns()-t)
                record['timings'].append({'lane':name,'mode':mode,'round':round_id,
                                         'order':order,'host_ns':samples})
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            batches[name].replay()
            end.record()
            end.synchronize()
            total_us = start.elapsed_time(end)*1000
            record['timings'].append({'lane':name,'mode':'device_batch','round':round_id,
                                     'order':order,'batch_us':total_us,
                                     'amortized_us':total_us/1000})
    cell['measurement'] = record
