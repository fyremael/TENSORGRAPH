"""Bounded recovery of the owned L4 session after pre-submission CLI failure.

No allocation occurs. The original dispatch/logs remain unchanged. Refuses to
run if the original controller has not finished or a recovery already exists.
"""
import json
from pathlib import Path
import subprocess
import zipfile

root=Path('/home/jim/gcl-research/qkv-performance-artifacts/L4')
original=json.loads((root/'dispatch.json').read_text())
assert original.get('in_progress') is None
assert original.get('runtime_retained') is True
assert original['session']=='tg-qkv-performance-l4-59210359'
assert original['commit']=='59210359907c47c73f95304cb17454558daa6f5a'
log=(root/'execute.log').read_text()
assert 'Connection was lost' in log and 'execution.py:217' in log
recovery=root/'recovery-1'
recovery.mkdir(exist_ok=False)
state={'session':original['session'],'commit':original['commit'],'statuses':{},'passed':False}
def call(label,args,timeout=120):
    state['in_progress']=label
    (recovery/'dispatch.json').write_text(json.dumps(state,indent=2))
    with (recovery/(label+'.log')).open('w') as handle:
        try: rc=subprocess.run(['colab',*args],stdout=handle,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired: rc=124
    state['statuses'][label]=rc
    state['in_progress']=None
    (recovery/'dispatch.json').write_text(json.dumps(state,indent=2))
    return rc
session=state['session']
rc=call('execute',['exec','-s',session,'-f',str(root/'bootstrap.py'),'--timeout','5400'],5460)
retrieved=call('download',['download','-s',session,'/content/qkv-performance-export.zip',str(root/'export.zip')],600)==0
call('notebook',['log','-s',session,'-o',str(recovery/'execution.ipynb')])
if retrieved:
    with zipfile.ZipFile(root/'export.zip') as z:
        process=json.loads(z.read('process-status.json'))
        if process['returncode']==0:
            d=json.loads(z.read('results/evidence.json'))
            state['passed']=d['passed'] and not d['dirty'] and d['commit']==state['commit']
if retrieved and rc!=124:
    call('stop',['stop','-s',session])
else:
    state['runtime_retained']=True
(recovery/'dispatch.json').write_text(json.dumps(state,indent=2))
