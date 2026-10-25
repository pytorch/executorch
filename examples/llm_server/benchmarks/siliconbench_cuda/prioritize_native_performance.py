"""Delay only an owned CPU case waiting for the GPU until performance finishes.

Kernel file-lock ownership and absent settings prove the server has not started.
Every exit restores the same stopped CPU process. Runtime source is unchanged.
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

from handoff_native_c1_quality import ParentHold, interrupted
from enforce_owned_tuning_deadline import identity


def lock_owner(path):
    st = path.stat()
    key = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    owners = []
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        if '->' in fields or len(fields) < 8 or fields[1] != 'FLOCK':
            continue
        major, minor, inode = fields[5].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == key:
            owners.append(int(fields[4]))
    assert len(owners) <= 1
    return owners[0] if owners else None


def self_test():
    import fcntl
    with tempfile.TemporaryDirectory(prefix='cpu-waiter-priority-') as temp:
        lock_path = Path(temp) / 'GPU.lock'
        marker = Path(temp) / 'acquired'
        owner = lock_path.open('a')
        fcntl.flock(owner, fcntl.LOCK_EX)
        assert lock_owner(lock_path) == os.getpid()
        waiter_code = ('import fcntl,pathlib; p=pathlib.Path(' + repr(str(lock_path))
                       + '); f=p.open("a");print("ready",flush=True);'
                       'fcntl.flock(f,fcntl.LOCK_EX);pathlib.Path('
                       + repr(str(marker)) + ').write_text("acquired")')
        waiter = subprocess.Popen([sys.executable, '-B', '-c', waiter_code],
                                  stdout=subprocess.PIPE, text=True)
        guard = None
        try:
            assert waiter.stdout.readline().strip() == 'ready'
            expected = identity(waiter.pid)
            assert expected
            code = ('import sys,time,signal;sys.path.insert(0,'
                    + repr(str(Path(__file__).resolve().parent)) + ');'
                    'from handoff_native_c1_quality import ParentHold,interrupted;'
                    'signal.signal(signal.SIGTERM,interrupted)\n'
                    'try:\n with ParentHold(' + str(waiter.pid) + ',' + repr(expected)
                    + '):\n  print("held",flush=True);time.sleep(60)\n'
                    'except InterruptedError:\n print("restored",flush=True)')
            guard = subprocess.Popen([sys.executable, '-B', '-c', code],
                                     stdout=subprocess.PIPE, text=True)
            assert guard.stdout.readline().strip() == 'held'
            assert lock_owner(lock_path) == os.getpid()
            fcntl.flock(owner, fcntl.LOCK_UN)
            probe_code = ('import fcntl,pathlib;f=pathlib.Path('
                          + repr(str(lock_path)) + ').open("a");'
                          'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
            subprocess.run([sys.executable, '-B', '-c', probe_code], check=True, timeout=5)
            assert identity(waiter.pid) == expected and not marker.exists()
            assert Path('/proc', str(waiter.pid), 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'T'
            guard.send_signal(signal.SIGTERM)
            assert guard.stdout.readline().strip() == 'restored'
            assert guard.wait(timeout=5) == 0 and waiter.wait(timeout=5) == 0
            assert marker.read_text() == 'acquired'
            return dict(passed=True, stopped_waiter_does_not_block_other_lock_owner=True,
                        external_SIGTERM_restores_waiter=True, real_Linux_flock_used=True,
                        no_GPU_requests=True)
        finally:
            for child in [guard, waiter]:
                if child is not None:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
            owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--case', type=Path)
    p.add_argument('--prioritized-matrix', type=Path)
    p.add_argument('--lock', type=Path)
    p.add_argument('--run-case-sha256')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-seconds', type=int, default=28800)
    a = p.parse_args()
    if a.self_test:
        print('TASK_JSON:' + json.dumps(self_test()))
        return
    assert all(v is not None for v in [a.case, a.prioritized_matrix, a.lock,
                                      a.run_case_sha256, a.output])
    assert 1 <= a.max_seconds <= 28800 and not a.output.exists()
    assert a.case.name == 'correctness-agent-vllm-prefix-on-c1'
    plan = json.loads((a.prioritized_matrix / 'matrix.json').read_text())
    expected_cases = {('performance', cache, c, backend)
                      for cache in ['off', 'on'] for c in [1, 2, 4, 8, 16]
                      for backend in ['vllm', 'sglang']}
    assert plan['model'] == 'muse-glimmer-30b'
    assert {tuple(row) for row in plan['cases']} == expected_cases
    a.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def save(status, **fields):
        temp = a.output / 'progress.pending.json'
        temp.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                       **fields), indent=2) + '\n')
        temp.replace(a.output / 'progress.json')

    def performance_complete():
        s = json.loads((a.prioritized_matrix / 'progress.json').read_text())
        assert s['status'] not in ['failed', 'cancelled', 'aborted']
        if s['status'] != 'completed':
            return False
        expected = {f'{phase}-agent-{backend}-prefix-{cache}-c{c}'
                    for phase, cache, c, backend in expected_cases}
        assert len(s['completed']) == 20 and set(s['completed']) == expected
        for name in expected:
            q = a.prioritized_matrix / name
            assert json.loads((q / 'progress.json').read_text())['status'] == 'completed'
            assert (q / 'memory-summary.json').exists()
        return True

    end = time.monotonic() + a.max_seconds
    try:
        save('waiting-owned-case-before-GPU-admission')
        while not performance_complete():
            assert time.monotonic() < end, 'Bounded priority wait expired'
            path = a.case / 'progress.json'
            if not path.exists():
                time.sleep(2)
                continue
            s = json.loads(path.read_text())
            pid = s['controller_pid']
            item = identity(pid)
            if s['status'] != 'waiting-GPU-lock' or (a.case / 'settings.json').exists():
                save('not-delayed-server-already-admitted', case_pid=pid, case_status=s['status'])
                return
            assert item
            raw = Path('/proc', str(pid), 'cmdline').read_bytes()
            args = [x.decode() for x in raw.split(b'\0') if x]
            script = next(Path(x) for x in args if x.endswith('/run_case_24h.py'))
            assert hashlib.sha256(script.read_bytes()).hexdigest() == a.run_case_sha256
            for flag, value in [('--mode', 'correctness'), ('--variant', 'vllm'),
                                ('--model', 'muse-glimmer-30b'), ('--prefix-cache', 'on'),
                                ('--concurrency', '1'), ('--output', str(a.case))]:
                assert args[args.index(flag) + 1] == value
            owner = lock_owner(a.lock)
            if owner is None:
                # Missing kernel ownership evidence cannot authorize a hold.
                time.sleep(2)
                continue
            if owner == pid:
                save('not-delayed-case-already-owns-GPU-lock', case_pid=pid)
                return
            with ParentHold(pid, item):
                if lock_owner(a.lock) == pid or (a.case / 'settings.json').exists():
                    save('not-delayed-case-acquired-lock-during-check', case_pid=pid)
                    return
                assert identity(pid) == item
                assert Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'T'
                save('holding-owned-CPU-waiter-for-native-performance', case_pid=pid,
                     case_start=item[1], case_command_SHA256=hashlib.sha256(raw).hexdigest(),
                     original_lock_owner=owner, server_not_started=True)
                while not performance_complete():
                    assert time.monotonic() < end, 'Bounded priority hold expired'
                    assert identity(pid) == item and not (a.case / 'settings.json').exists()
                    assert lock_owner(a.lock) != pid, 'Held CPU waiter acquired lock; restore it'
                    time.sleep(5)
            save('completed', all20_native_performance_cells_complete=True,
                 same_CPU_waiter_resumed=True, case_pid=pid, no_inference_cancelled=True)
            return
        save('completed', performance_already_complete=True, no_case_delayed=True)
    except BaseException as exc:
        save('failed', error_type=type(exc).__name__, detail=str(exc)[-1000:],
             stopped_CPU_waiter_restored=True)
        raise


if __name__ == '__main__':
    main()
