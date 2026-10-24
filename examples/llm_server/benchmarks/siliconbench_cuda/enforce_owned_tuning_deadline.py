"""Enforce a tuning deadline on exact registered services and descendants.

Configuration hashes bind ownership. Freeze before discovering descendants so
workers in independent sessions cannot outlive their stopped controller.
Main performance and correctness services are explicitly excluded.
"""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

def identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ['Z','X'] else (pid,int(fields[19]))
    except FileNotFoundError:
        return None

def send(item,sig):
    if identity(item[0])==item:
        try:os.kill(item[0],sig)
        except ProcessLookupError:pass

def stop_owned_tree(pid,stop_root,gpu_wait=None):
    frozen=[]
    def freeze(p):
        item=identity(p)
        if item is None:return
        send(item,signal.SIGSTOP)
        end=time.monotonic()+5
        while identity(p)==item:
            if Path('/proc',str(p),'stat').read_text().rsplit(')',1)[1].split()[0]=='T':break
            assert time.monotonic()<end,'Cannot safely freeze the owned process'
            time.sleep(.02)
        if identity(p)!=item:return
        frozen.append(item)
        children=set()
        for thread in Path('/proc',str(p),'task').iterdir():
            children.update(thread.joinpath('children').read_text().split())
        assert len(frozen)<512,'Unexpectedly large owned process tree'
        for child in children:freeze(int(child))
    try:
        freeze(pid)
        descendants=[x for x in frozen if x[0]!=pid]
        for item in reversed(descendants):
            send(item,signal.SIGTERM);send(item,signal.SIGCONT)
        end=time.monotonic()+5
        while any(identity(p)==item for item in descendants for p in [item[0]]) and time.monotonic()<end:
            time.sleep(.05)
        for item in reversed(descendants):send(item,signal.SIGKILL)
        if gpu_wait:gpu_wait([item[0] for item in descendants])
        stop_root()
        if gpu_wait:gpu_wait([item[0] for item in frozen])
        end=time.monotonic()+10
        while any(identity(item[0])==item for item in frozen) and time.monotonic()<end:
            time.sleep(.05)
        assert all(identity(item[0])!=item for item in frozen),'Owned process remained after cutoff'
        return dict(root_pid=pid,frozen_start_time_identities=frozen,independent_session_descendants_stopped=True)
    except BaseException:
        for item in frozen:send(item,signal.SIGCONT)
        raise

def self_test():
    sleeper='import time;time.sleep(600)'
    root_code='import subprocess,sys,json,time;p=subprocess.Popen([sys.executable,"-c",'+repr(sleeper)+'],start_new_session=True);print(json.dumps({"child":p.pid}),flush=True);time.sleep(600)'
    unrelated=subprocess.Popen([sys.executable,'-c',sleeper],start_new_session=True)
    root=subprocess.Popen([sys.executable,'-c',root_code],start_new_session=True,stdout=subprocess.PIPE,text=True)
    try:
        child=json.loads(root.stdout.readline())['child'];root_id=identity(root.pid);other_id=identity(unrelated.pid)
        assert root_id and identity(child) and other_id
        result=stop_owned_tree(root.pid,lambda:send(root_id,signal.SIGKILL))
        root.wait(timeout=10)
        assert identity(child) is None and identity(unrelated.pid)==other_id
        return dict(passed=True,real_Linux_independent_session_child_terminated=True,unrelated_process_preserved=True,tree=result)
    finally:
        for process in [root,unrelated]:
            if process.poll() is None:process.kill()
            process.wait(timeout=10)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    if a.self_test:
        (a.output/'results.json').write_text(json.dumps(self_test(),indent=2)+'\n');return
    assert a.plan is not None
    plan=json.loads(a.plan.read_text());assert plan['main_and_quality_excluded']
    for row in plan['programs']:
        name=row['program'];assert name.startswith('sb5090') and row['task_kind']=='tuning'
        assert not any(x in name.lower() for x in ['-main','-quality','host-calibration'])
        assert len(row['config_sha256'])==64
    deadline=datetime.datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00'))
    def save(status,**fields):
        pending=a.output/'progress.pending.json';pending.write_text(json.dumps(dict(status=status,deadline=plan['deadline_utc'],**fields),indent=2)+'\n');pending.replace(a.output/'progress.json')
    save('waiting-user-tuning-cutoff',controller_pid=os.getpid())
    while True:
        remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if remaining<=0:break
        time.sleep(min(remaining,30))
    results=[]
    for row in plan['programs']:
        name=row['program'];config=Path('/etc/supervisor/conf.d',name+'.conf')
        if not config.exists() or hashlib.sha256(config.read_bytes()).hexdigest()!=row['config_sha256']:
            results.append(dict(program=name,action='configuration-changed-left-untouched'));continue
        raw=subprocess.run(['supervisorctl','pid',name],capture_output=True,text=True,timeout=30).stdout.strip()
        pid=int(raw) if raw.isdigit() else 0
        if not pid or identity(pid) is None:
            results.append(dict(program=name,action='already-terminal'));continue
        def stop():
            subprocess.run(['supervisorctl','stop',name],capture_output=True,text=True,check=True,timeout=120)
        def wait_gpu(owned):
            end=time.monotonic()+30
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20)
                current={int(s.strip()) for s in raw.splitlines() if s.strip().isdigit()}
                if not current.intersection(owned):return
                assert time.monotonic()<end,'Owned CUDA context teardown did not finish'
                time.sleep(.5)
        result=stop_owned_tree(pid,stop,wait_gpu)
        results.append(dict(program=name,action='user-tuning-cutoff-stop',**result))
    save('completed',results=results,main_and_quality_not_targeted=True)

if __name__=='__main__':main()
