"""Wait for an owned experiment checkpoint, then run a reviewed command.

Only observe existing work. Do not restart or stop it. The optional shared GPU
lock is held through the entire follow-up command, including CPU preparation.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--progress',type=Path,required=True)
parser.add_argument('--accepted',nargs='+',default=['completed'])
parser.add_argument('--lock',type=Path)
parser.add_argument('--poll-seconds',type=float,default=30)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command and args.command[0]=='--' else args.command
assert command and Path(command[0]).is_absolute() and args.progress.exists()
assert 5<=args.poll_seconds<=60
while True:
    state=json.loads(args.progress.read_text())
    if state['status'] in args.accepted:break
    assert state['status'] not in ['failed','cancelled','aborted'],state['status']
    time.sleep(args.poll_seconds)
if args.lock:
    with args.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        deadline=time.monotonic()+120
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            assert time.monotonic()<deadline,'GPU remains occupied; leave existing processes untouched'
            time.sleep(5)
        subprocess.run(command,check=True)
else:
    subprocess.run(command,check=True)
