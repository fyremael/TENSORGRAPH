import hashlib,itertools,json,math,statistics,zipfile

def analyze(gpu):
 p='/home/jim/gcl-research/qkv-performance-artifacts/'+gpu+'/export.zip'
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  d=json.loads(z.read('results/evidence.json'))
  assert json.loads(z.read('process-status.json'))['returncode']==0
  assert d['passed'] and not d['dirty'] and d['commit']=='59210359907c47c73f95304cb17454558daa6f5a'
  assert {(c['dtype'],c['batch'],c['hidden']) for c in d['cells']}==set(itertools.product(('float16','bfloat16'),(1,4),(1024,4096)))
  lanes=('whole_eager_library','whole_native_library','whole_inductor','projection_library','projection_inductor')
  modes=('ordinary_host','graph_host','device_batch')
  result={'gpu':d['gpu'],'commit':d['commit'],'archive_path':p,'archive_sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'precision':{'tf32':d['tf32'],'reduced_precision_reduction':d['reduced_precision_reduction']},'cells':[],'archive_crc_and_source_check':True}
  for name in z.namelist():
   if name.endswith('/artifact.json'):
    a=json.loads(z.read(name));assert hashlib.sha256(a['source'].encode()).hexdigest()==a['source_sha256']
    assert a['source'].strip() in z.read(name.replace('artifact.json','cuda.cu')).decode()
  for c in d['cells']:
   assert c['passed'] and all(l['passed'] and l['ordinary_checks']==5 and l['replay_checks']==100 for l in c['lanes'])
   m=c['measurement'];ts=m['timings']
   assert len(ts)==75 and {(r['lane'],r['mode'],r['round']) for r in ts}==set(itertools.product(lanes,modes,range(5)))
   item={k:c[k] for k in ('dtype','batch','hidden')};item.update(allocation={},timing={})
   for key,a in m['allocation'].items():
    trace=json.loads(z.read('results/'+a['trace']))['traceEvents']
    allocs=[e for e in trace if e.get('name')=='[memory]' and e.get('args',{}).get('Device Type') in (1,'CUDA') and e.get('args',{}).get('Bytes',0)>0]
    apis=[e['name'] for e in trace if e.get('name','').startswith(('cudaMalloc','cuMemAlloc')) and 'Host' not in e['name']]
    item['allocation'][key]={'peak_extra_bytes':a['peak_extra_bytes'],'positive_cuda_memory_events':len(allocs),'positive_cuda_bytes':sum(e['args']['Bytes'] for e in allocs),'allocation_api_events':apis,'profiled_calls':a['profiled_calls']}
   for lane in lanes:
    item['timing'][lane]={}
    for mode in modes:
     records=[r for r in ts if r['lane']==lane and r['mode']==mode]
     if mode=='device_batch':
      assert len(records)==5
      values=[r['amortized_us'] for r in records]
     else:
      assert all(len(r['host_ns'])==1000 for r in records)
      values=[v/1000 for r in records for v in r['host_ns']]
     assert all(math.isfinite(v) and v>0 for v in values)
     ordered=sorted(values)
     item['timing'][lane][mode]={'median_us':statistics.median(values),'samples':len(values)}
     if mode!='device_batch':item['timing'][lane][mode].update(p95_us=ordered[int(.95*(len(values)-1))],p99_us=ordered[int(.99*(len(values)-1))])
   result['cells'].append(item)
  result['aggregate']={mode:{lane:statistics.median(c['timing'][lane][mode]['median_us'] for c in result['cells']) for lane in lanes} for mode in modes}
  result['ordinary_allocation_cases']={lane:sum(bool(c['allocation'][lane+'-ordinary']['peak_extra_bytes'] or c['allocation'][lane+'-ordinary']['positive_cuda_memory_events'] or c['allocation'][lane+'-ordinary']['allocation_api_events']) for c in result['cells']) for lane in lanes}
  result['replay_allocation_cases']={lane:sum(bool(c['allocation'][lane+'-replay']['peak_extra_bytes'] or c['allocation'][lane+'-replay']['positive_cuda_memory_events'] or c['allocation'][lane+'-replay']['allocation_api_events']) for c in result['cells']) for lane in lanes}
  return result

if __name__=='__main__':
 import sys
 print(json.dumps(analyze(sys.argv[1]),separators=(',',':')))
