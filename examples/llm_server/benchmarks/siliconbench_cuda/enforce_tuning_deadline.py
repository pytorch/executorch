"""Stop only registered task-owned tuning services at a user-defined cutoff.

Main comparison and correctness collection are excluded. Commands are bound to
reviewed Supervisor configuration hashes so a reused service is left untouched.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
plan=json.loads(a.plan.read_text())
deadline=datetime.datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00'))
for row in plan['programs']:
    name=row['program']
    assert name.startswith('sb5090')
    assert all(marker not in name for marker in ['-main','-native-','host-calibration'])
(a.output/'progress.json').write_text(json.dumps(dict(status='waiting-user-tuning-cutoff',deadline=plan['deadline_utc']))+'\n')
while True:
    remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining<=0:break
    time.sleep(min(remaining,30))
results=[]
for row in plan['programs']:
    name=row['program'];config=Path('/etc/supervisor/conf.d')/(name+'.conf')
    if not config.exists() or hashlib.sha256(config.read_bytes()).hexdigest()!=row['config_sha256']:
        results.append(dict(program=name,action='configuration-changed-left-untouched'))
        continue
    status=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True,timeout=30).stdout.split()
    if any(value in status for value in ['RUNNING','STARTING']):
        stopped=subprocess.run(['supervisorctl','stop',name],capture_output=True,text=True,timeout=120)
        results.append(dict(program=name,action='user-tuning-cutoff-stop',returncode=stopped.returncode))
    else:
        results.append(dict(program=name,action='already-terminal'))
(a.output/'progress.json').write_text(json.dumps(dict(status='completed',deadline=plan['deadline_utc'],results=results,main_comparison_not_targeted=True),indent=2)+'\n')
