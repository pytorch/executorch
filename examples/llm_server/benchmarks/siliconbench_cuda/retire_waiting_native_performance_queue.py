"""Retire a verified zero-request CPU queue before disjoint backend sharding.

One completed four-round cell stays intact. Current inference and other queues
are untouched. Both CPU identities, absent server settings, and actual kernel
file-lock ownership must be verified before either queued process is retired.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from enforce_owned_tuning_deadline import identity
from handoff_native_c1_quality import ParentHold, interrupted
from prioritize_native_performance import lock_owner, wait_stopped


def retire_pair(parent, parent_identity, child, child_identity, verify, save):
    with ParentHold(parent, parent_identity) as ph:
        wait_stopped(parent, parent_identity)
        with ParentHold(child, child_identity) as ch:
            wait_stopped(child, child_identity)
            verify()
            save('verified-both-owned-CPU-waiters-held')
            ch.stop_after_child_finished()
            assert identity(child) is None
            save('verified-zero-request-child-terminal')
        ph.stop_after_child_finished()
        assert identity(parent) is None
    save('verified-original-CPU-parent-terminal')


def self_test():
    import fcntl
    with tempfile.TemporaryDirectory(prefix='native-performance-shard-safety-') as temp:
        p = Path(temp)
        lock = p / 'GPU.lock'
        acquired = p / 'acquired'
        next_case = p / 'next-case'
        ticks = p / 'independent-ticks'
        with lock.open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            child_code = ('import fcntl,pathlib;f=pathlib.Path(' + repr(str(lock))
                          + ').open("a");fcntl.flock(f,fcntl.LOCK_EX);pathlib.Path('
                          + repr(str(acquired)) + ').write_text("acquired")')
            parent_code = ('import pathlib,subprocess,sys,time;p=subprocess.Popen('
                           + repr([sys.executable, '-B', '-c', child_code])
                           + ');print(p.pid,flush=True);r=p.wait();'
                           'pathlib.Path(' + repr(str(next_case))
                           + ').write_text("next") if r==0 else None;time.sleep(60)')
            parent = subprocess.Popen([sys.executable, '-B', '-c', parent_code],
                                      stdout=subprocess.PIPE, text=True)
            independent = subprocess.Popen([sys.executable, '-B', '-c',
                'import pathlib,time;p=pathlib.Path(' + repr(str(ticks))
                + ');n=0\nwhile True:\n n+=1;p.write_text(str(n));time.sleep(.03)'],
                start_new_session=True)
            child = None
            try:
                child = int(parent.stdout.readline().strip())
                parent_id, child_id = identity(parent.pid), identity(child)
                assert parent_id and child_id
                end = time.monotonic() + 3
                while not ticks.exists():
                    assert time.monotonic() < end
                    time.sleep(.03)
                before = int(ticks.read_text())

                def verify():
                    assert lock_owner(lock) == os.getpid()
                    assert not acquired.exists() and not next_case.exists()

                retire_pair(parent.pid, parent_id, child, child_id, verify,
                            lambda status: None)
                assert parent.wait(timeout=5) != 0
                time.sleep(.1)
                assert identity(independent.pid) and int(ticks.read_text()) > before
                verify()
                return dict(passed=True,real_Linux_flock_used=True,
                            other_lock_owner_unchanged=True,
                            independent_work_continued=True,
                            zero_request_child_and_queue_terminal=True,
                            no_next_case_started=True,no_GPU_requests=True)
            finally:
                for process in [parent, independent]:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                if child and identity(child):
                    os.kill(child, signal.SIGTERM)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--matrix', type=Path)
    p.add_argument('--parent-pid', type=int)
    p.add_argument('--parent-start', type=int)
    p.add_argument('--parent-command-sha256')
    p.add_argument('--parent-script-sha256')
    p.add_argument('--child-pid', type=int)
    p.add_argument('--child-start', type=int)
    p.add_argument('--child-command-sha256')
    p.add_argument('--child-script-sha256')
    p.add_argument('--C-plan', dest='c_plan', type=Path)
    p.add_argument('--F-plan', dest='f_plan', type=Path)
    p.add_argument('--lock', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.self_test:
        print('TASK_JSON:' + json.dumps(self_test()))
        return
    assert all(v is not None for k,v in vars(a).items() if k != 'self_test')
    assert not a.output.exists()
    original = json.loads((a.matrix / 'matrix.json').read_text())
    c_plan = json.loads(a.c_plan.read_text())
    f_plan = json.loads(a.f_plan.read_text())
    assert original['model'] == c_plan['model'] == f_plan['model'] == 'muse-glimmer-30b'
    expected = {('performance',cache,c,backend) for cache in ['off','on']
                for c in [1,2,4,8,16] for backend in ['vllm','sglang']}
    retained = {('performance','off',1,'vllm')}
    c_set, f_set = {tuple(row) for row in c_plan['cases']}, {tuple(row) for row in f_plan['cases']}
    assert {tuple(row) for row in original['cases']} == expected
    assert c_set == {r for r in expected if r[3]=='vllm'} - retained
    assert f_set == {r for r in expected if r[3]=='sglang'}
    assert not (c_set & f_set or c_set & retained or f_set & retained)
    assert c_set | f_set | retained == expected and len(c_set)==9 and len(f_set)==10
    state = json.loads((a.matrix / 'progress.json').read_text())
    assert state['status']=='running' and state['controller_pid']==a.parent_pid
    assert state['completed']==['performance-agent-vllm-prefix-off-c1']
    assert state['current_case']=='performance-agent-sglang-prefix-off-c1'
    case = a.matrix / state['current_case']
    completed = a.matrix / state['completed'][0]
    cs = json.loads((completed/'progress.json').read_text())
    assert cs['status']=='completed' and identity(cs['controller_pid']) is None
    settings = json.loads((completed/'settings.json').read_text())
    assert settings['mode']=='performance' and settings['variant']=='vllm'
    assert settings['concurrency']==1 and not settings['prefix_cache']
    assert (completed/'memory-summary.json').exists()
    for round_id in [1,2,3,4]:
        assert (completed/f'repeat-{round_id}/metrics.json').exists()
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in completed.rglob('*') if path.is_file() and path.suffix in ['.json','.jsonl']}
    parent_id, child_id = (a.parent_pid,a.parent_start),(a.child_pid,a.child_start)

    def verify():
        assert identity(a.parent_pid)==parent_id and identity(a.child_pid)==child_id
        for pid, cmd_sha, script_sha, suffix in [
            (a.parent_pid,a.parent_command_sha256,a.parent_script_sha256,'/run_matrix_24h.py'),
            (a.child_pid,a.child_command_sha256,a.child_script_sha256,'/run_case_24h.py')]:
            raw = Path('/proc',str(pid),'cmdline').read_bytes()
            assert hashlib.sha256(raw).hexdigest()==cmd_sha
            args = [x.decode() for x in raw.split(b'\0') if x]
            script = next(Path(x) for x in args if x.endswith(suffix))
            assert hashlib.sha256(script.read_bytes()).hexdigest()==script_sha
            assert args[args.index('--output')+1]==str(a.matrix if pid==a.parent_pid else case)
            assert args[args.index('--model')+1]=='muse-glimmer-30b'
            if pid==a.child_pid:
                for flag,value in [('--mode','performance'),('--variant','sglang'),
                                    ('--prefix-cache','off'),('--concurrency','1')]:
                    assert args[args.index(flag)+1]==value
        child_stat = Path('/proc',str(a.child_pid),'stat').read_text().rsplit(')',1)[1].split()
        assert int(child_stat[1])==a.parent_pid
        cp = json.loads((case/'progress.json').read_text())
        assert cp['status']=='waiting-GPU-lock' and cp['controller_pid']==a.child_pid
        assert not (case/'settings.json').exists() and not (case/'server.log').exists()
        owner = lock_owner(a.lock)
        assert owner is not None and owner not in [a.parent_pid,a.child_pid]

    verify()
    a.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)

    def save(status, **fields):
        temporary = a.output / 'progress.pending.json'
        temporary.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields),indent=2)+'\n')
        temporary.replace(a.output/'progress.json')

    try:
        save('verified-owned-zero-request-performance-queue')
        retire_pair(a.parent_pid,parent_id,a.child_pid,child_id,verify,save)
        assert not (case/'settings.json').exists() and not (case/'server.log').exists()
        for path,sha in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        record=dict(original_CPU_parent_terminal=True,original_zero_request_child_terminal=True,
                    retained_complete_cell_preserved=True,retained_cell=str(completed),
                    retained_response_metric_SHA256=hashes,
                    retired_zero_request_case=str(case),server_never_started=True,
                    original_parent_identity=parent_id,original_child_identity=child_id,
                    C_plan=str(a.c_plan),F_plan=str(a.f_plan),
                    C_plan_SHA256=hashlib.sha256(a.c_plan.read_bytes()).hexdigest(),
                    F_plan_SHA256=hashlib.sha256(a.f_plan.read_bytes()).hexdigest(),
                    retained_case_count=1,C_remaining_cases=9,F_remaining_cases=10,
                    all20_original_cases_exactly_preserved=True,no_GPU_inference_interrupted=True)
        marker=a.matrix/'retirement-disjoint-performance-handoff.json'
        assert not marker.exists()
        temporary=marker.with_suffix('.pending.json')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(marker)
        save('completed',**record)
        print('TASK_JSON:'+json.dumps(record))
    except BaseException as exc:
        save('failed',error_type=type(exc).__name__,detail=str(exc)[-1000:],
             parent_live=identity(a.parent_pid) is not None,child_live=identity(a.child_pid) is not None)
        raise


if __name__=='__main__':
    main()
