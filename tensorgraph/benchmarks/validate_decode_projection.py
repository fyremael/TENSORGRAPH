"""Numerical GPU validation of the mixed-backend residual/RMSNorm/QKV subgraph."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    import torch
    import triton
    from tensorgraph.codegen.decode import DecodeGraph, compile_decode
    from tensorgraph.benchmarks.decode_projection import make_projection_lanes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['A100', 'L4'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    def shell(*argv):
        return subprocess.check_output(argv, text=True).strip()
    assert args.gpu in torch.cuda.get_device_name()
    assert torch.__version__ == '2.11.0+cu130'
    assert torch.version.cuda == '13.0' and triton.__version__ == '3.6.0'
    nvcc = shell('nvcc', '--version')
    assert 'V13.0.88' in nvcc
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(731)
    doc = {'schema': 'TG-QKV-VALIDATION/1', 'commit': shell('git', 'rev-parse', 'HEAD'),
           'dirty': bool(shell('git', 'status', '--porcelain')), 'gpu': torch.cuda.get_device_name(),
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__,
           'nvcc': nvcc, 'seed': 731, 'tf32': False, 'reduced_precision_reduction': False,
           'scope': 'generated native normalization plus library GEMM; not generated GEMM',
           'tolerances': {'float16': {'norm': [0.002, 0.002], 'qkv': [0.02, 0.02]},
                          'bfloat16': {'norm': [0.02, 0.02], 'qkv': [0.0625, 0.04]}},
           'oracle': 'CPU FP64 normalization, explicit storage cast, CUDA FP32 projection with TF32 disabled',
           'cells': [], 'passed': False}
    def save():
        (out/'evidence.json').write_text(json.dumps(doc, indent=2, allow_nan=False))
    save()
    for dt in ('float16', 'bfloat16'):
        dtype = getattr(torch, dt)
        for h in (1024, 4096):
            artifact = DecodeGraph(dt, h).emit()
            build = out/f'build-{dt}-{h}'
            build.mkdir()
            (build/'artifact.json').write_text(json.dumps(artifact, indent=2))
            native = compile_decode(artifact, build)
            for batch in (1, 4):
                x = torch.randn(batch,h,device='cuda',dtype=dtype)
                residual = torch.randn_like(x)
                weight = torch.randn(h,device='cuda',dtype=dtype)
                qweight = (torch.randn(h,3*h,device='cuda')/(h**0.5)).to(dtype)
                wf = qweight.float()
                torch._dynamo.reset()
                state = make_projection_lanes(torch,native,x,residual,weight,qweight)
                outputs = state['outputs']
                inputs = (x,residual,weight,qweight)
                ptrs = [t.data_ptr() for t in inputs+outputs]
                cell = {'dtype':dt,'batch':batch,'hidden':h,'lanes':[],
                        'source_sha256': artifact['source_sha256']}
                doc['cells'].append(cell)
                for name,fn in state['lanes'].items():
                    t0 = time.perf_counter_ns()
                    fn()
                    torch.cuda.synchronize()
                    lane = {'name':name,'first_execution_ns':time.perf_counter_ns()-t0,
                            'ordinary_checks':0,'replay_checks':0,'max_abs':[0.,0.,0.]}
                    cell['lanes'].append(lane)
                    def check(call,label):
                        before = tuple(t.clone() for t in inputs[:3])
                        # Full projection weight is immutable and checked once per lane below.
                        call()
                        s = x.cpu().double()+residual.cpu().double()
                        n = (s*torch.rsqrt((s*s).mean(-1,keepdim=True)+1e-5)*weight.cpu().double()).to(dtype).cuda()
                        expected = (s.to(dtype).cuda(),n,(n.float()@wf).to(dtype))
                        try:
                            for i,(got,ref) in enumerate(zip(outputs,expected)):
                                atol,rtol=doc['tolerances'][dt]['norm' if i<2 else 'qkv']
                                torch.testing.assert_close(got,ref,atol=atol,rtol=rtol,msg=label)
                                lane['max_abs'][i]=max(lane['max_abs'][i],float((got.float()-ref.float()).abs().max()))
                            assert all(torch.equal(a,b) for a,b in zip(inputs[:3],before)), 'input changed'
                            assert ptrs==[a.data_ptr() for a in inputs+outputs], 'pointer changed'
                        except Exception:
                            torch.save({'inputs':[a.cpu() for a in inputs[:3]],
                                        'projection_weight':qweight.cpu(),
                                        'outputs':[a.cpu() for a in outputs],
                                        'expected':[a.cpu() for a in expected]},out/'failed-values.pt')
                            lane['failed_label']=label
                            save()
                            raise
                    checksum_before = hashlib.sha256(qweight.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for case in ('normal','zero','nearzero','cancellation','large'):
                        x.normal_();residual.normal_();weight.normal_()
                        if case=='zero': x.zero_();residual.zero_()
                        if case=='nearzero': x.mul_(1e-7);residual.mul_(1e-7)
                        if case=='cancellation': residual.copy_(-x)
                        if case=='large': x.fill_(100);residual.fill_(100)
                        check(fn,name+':'+case)
                        lane['ordinary_checks']+=1
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3): fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): fn()
                    for i in range(100):
                        x.normal_();residual.normal_()
                        if i%10==0: residual.copy_(-x)
                        check(graph.replay,name+':replay'+str(i))
                        lane['replay_checks']+=1
                    assert checksum_before==hashlib.sha256(qweight.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(), 'projection weight changed'
                    lane['projection_weight_unchanged']=True
                    lane['passed']=True
                    save()
                cell['passed']=True
                save()
                print('CELL PASSED',dt,batch,h,flush=True)
                del state, outputs, inputs, wf, qweight, graph
                torch.cuda.empty_cache()
    doc['passed']=len(doc['cells'])==8 and all(c.get('passed') for c in doc['cells'])
    save()


if __name__=='__main__':
    main()
