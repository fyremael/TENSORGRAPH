#!/usr/bin/env python3
"""Serial A100/L4 execution using Colab CLI; preserve evidence before cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path(__file__).resolve().parents[1]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
output=root.parent/'wp03b-artifacts'
output.mkdir(exist_ok=False)
for gpu in ('A100','L4'):
    session='tg-wp03b-'+gpu.lower()+'-'+commit[:8]
    dest=output/gpu;dest.mkdir()
    state={'session':session,'gpu':gpu,'commit':commit,'statuses':{},'passed':False}
    def call(label,argv,timeout=120):
        state['in_progress']=label
        (dest/'dispatch.json').write_text(json.dumps(state,indent=2))
        with (dest/(label+'.log')).open('w') as log:
            try: rc=subprocess.run(['colab',*argv],stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired: rc=124
        state['statuses'][label]=rc
        state['in_progress']=None
        (dest/'dispatch.json').write_text(json.dumps(state,indent=2))
        return rc
    if call('allocate',['new','-s',session,'--gpu',gpu]):
        call('partial_cleanup',['stop','-s',session]);continue
    bootstrap=dest/'bootstrap.py'
    bootstrap.write_text(f'''import os, pathlib, subprocess, sys, shutil, json, traceback
base=pathlib.Path('/content/wp03b')
base.mkdir(exist_ok=True)
status={{'returncode':1,'commit':{commit!r}}}
try:
    subprocess.run(['git','clone','https://github.com/fyremael/TENSORGRAPH.git','/content/tensorgraph'],check=True)
    subprocess.run(['git','checkout',{commit!r}],cwd='/content/tensorgraph',check=True)
    import torch
    if torch.__version__ != '2.11.0+cu130':
        raise RuntimeError('Unexpected PyTorch build: '+torch.__version__)
    env=dict(os.environ,PYTHONPATH='/content/tensorgraph',MAX_JOBS='2',TORCHINDUCTOR_CACHE_DIR='/content/wp03b/inductor-cache')
    with (base/'worker.log').open('w') as log:
        p=subprocess.run([sys.executable,'-m','tensorgraph.benchmarks.decode','--gpu',{gpu!r},'--output','/content/wp03b/results'],cwd='/content/tensorgraph',env=env,stdout=log,stderr=subprocess.STDOUT)
    status['returncode']=p.returncode
except Exception:
    (base/'exception.txt').write_text(traceback.format_exc())
finally:
    (base/'process-status.json').write_text(json.dumps(status))
    shutil.make_archive('/content/wp03b-export','zip',base)
if status['returncode']: raise RuntimeError('WP03-B failed; see retained export')
''')
    rc=call('execute',['exec','-s',session,'-f',str(bootstrap),'--timeout','5400'],5460)
    retrieved=call('download',['download','-s',session,'/content/wp03b-export.zip',str(dest/'export.zip')],600)==0
    call('notebook',['log','-s',session,'-o',str(dest/'execution.ipynb')])
    if retrieved:
        with zipfile.ZipFile(dest/'export.zip') as z:
            p=json.loads(z.read('process-status.json'))
            if p['returncode']==0:
                evidence=json.loads(z.read('results/evidence.json'))
                state['passed']=evidence.get('passed') is True and evidence['commit']==commit and not evidence['dirty']
    if retrieved and rc!=124:
        call('stop',['stop','-s',session])
    else:
        state['runtime_retained']=True
        (dest/'dispatch.json').write_text(json.dumps(state,indent=2));break
    (dest/'dispatch.json').write_text(json.dumps(state,indent=2))
subprocess.run(['colab','sessions'])
