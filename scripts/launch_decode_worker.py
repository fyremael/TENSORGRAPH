"""Run with Colab exec; starts one worker and returns its PID, never a pass claim.

Requires explicit TG_SOURCE_COMMIT and TG_EXPECT_GPU env settings. Download
/content/wp03b-export.zip and inspect process-status.json and evidence.json
before declaring completion or releasing the runtime.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

commit = os.environ["TG_SOURCE_COMMIT"]
gpu = os.environ["TG_EXPECT_GPU"]
assert re.fullmatch("[0-9a-f]{40}", commit)
assert gpu in ("A100", "L4")
base = Path("/content/wp03b")
base.mkdir(exist_ok=False)
subprocess.run(
    ["git", "clone", "https://github.com/fyremael/TENSORGRAPH.git", "/content/tensorgraph"],
    check=True,
)
subprocess.run(["git", "checkout", commit], cwd="/content/tensorgraph", check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "ninja"], check=True)
env = dict(
    os.environ,
    PYTHONPATH="/content/tensorgraph",
    MAX_JOBS="2",
    TORCHINDUCTOR_CACHE_DIR="/content/wp03b/inductor-cache",
)
# The child retains failure evidence and packages it even if the benchmark raises.
worker = """import json,pathlib,shutil,subprocess,sys,traceback
base=pathlib.Path('/content/wp03b')
status={'returncode':1,'commit':sys.argv[1]}
try:
    p=subprocess.run([sys.executable,'-u','-m','tensorgraph.benchmarks.decode','--gpu',sys.argv[2],'--output',str(base/'results')],cwd='/content/tensorgraph')
    status['returncode']=p.returncode
except Exception:
    (base/'exception.txt').write_text(traceback.format_exc())
finally:
    (base/'process-status.json').write_text(json.dumps(status))
    shutil.make_archive('/content/wp03b-export','zip',base)
"""
with (base / "worker.log").open("x") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", worker, commit, gpu],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(base / "launch.json").write_text(
    __import__("json").dumps({"pid": process.pid, "commit": commit, "gpu": gpu})
)
print("Worker launched:", process.pid, "Result not yet known.", flush=True)
