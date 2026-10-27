"""Profile an existing, hash-pinned ET wrapper without transferring artifacts.

Separate unprofiled controls from Nsight data. Existing model/runtime bindings,
request files, and accepted results remain unchanged. Fixed-output control and
causal attribution are still required after collecting these HTTP traces.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['wrapper', 'inputs', 'client', 'python', 'benchmark-python',
                 'output', 'lock', 'binding-receipt']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--after-progress', type=Path)
    p.add_argument('--phase', choices=['collect', 'serve', 'profile'], default='collect')
    p.add_argument('--port', type=int, default=19637)
    a = p.parse_args()
    pkg = Path(__file__).resolve().parent.parent
    receipt = json.loads(a.binding_receipt.read_text())
    assert sha(a.wrapper) == receipt['wrapper_SHA256']
    assert sha(a.client) == receipt['client_SHA256']
    for split in ['chat', 'agent']:
        assert sha(a.inputs / (split + '_benchmark_prompts.json')) == receipt[split + '_prompts_SHA256']
    assert receipt['prefill_budget'] == 2048 and receipt['unchanged_existing_main_wrapper']
    command = ['bash', str(a.wrapper), 'executorch']
    if a.phase == 'serve':
        os.execvpe(command[0], command, os.environ)
    if a.phase == 'profile':
        case = Path(os.environ['SCALING_PROFILE_CASE'])
        argv = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none',
                '--cpuctxsw=none', '--cuda-graph-trace=graph',
                '--cuda-flush-interval=1000', '--wait=all', '--output',
                str(case / 'graph-trace'), *command]
        child = subprocess.Popen(argv, start_new_session=True)

        def stop(*_):
            found = []
            for d in Path('/proc').iterdir():
                if not d.name.isdigit():
                    continue
                try:
                    cmd = (d / 'cmdline').read_bytes().split(b'\0')
                    stat = (d / 'stat').read_text().rsplit(') ', 1)[1].split()
                    server = any(x.endswith(b'/serve_gemma_runtime_columns.py') for x in cmd)
                    if int(stat[3]) == child.pid and server:
                        found.append(int(d.name))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            for pid in found:
                os.kill(pid, signal.SIGINT)
            if not found:
                os.killpg(child.pid, signal.SIGINT)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        rc = child.wait()
        (case / 'profile-finalization.json').write_text(json.dumps(dict(
            returncode=rc, argv=argv, graceful_shutdown_requested=True,
            profiler_kept_alive_to_flush=True), indent=2) + '\n')
        sys.exit(0 if rc == 0 else 1)

    a.output.mkdir(parents=True, exist_ok=False)
    completed = []

    def state(status, **fields):
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, PID=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            main_remains_paused=True, completed=completed, **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')

    try:
        if a.after_progress:
            state('waiting-for-existing-mini-export-to-exit')
            deadline = time.monotonic() + 10800
            while True:
                r = json.loads(a.after_progress.read_text())
                if r['status'] in ['real-mini-artifacts-exported', 'failed-real-mini-export']:
                    break
                assert time.monotonic() < deadline, 'Previous export wait exceeded bound'
                time.sleep(20)
            deadline = time.monotonic() + 60
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip():
                assert time.monotonic() < deadline, 'Previous GPU context has not exited'
                time.sleep(1)
        common = ['--wrapper', str(a.wrapper), '--inputs', str(a.inputs),
                  '--client', str(a.client), '--python', str(a.python),
                  '--benchmark-python', str(a.benchmark_python), '--output',
                  str(a.output), '--lock', str(a.lock), '--binding-receipt',
                  str(a.binding_receipt), '--port', str(a.port)]
        for profile in [False, True]:
            for split, prefix in [('chat', 'on'), ('agent', 'off')]:
                for c in [8, 16]:
                    case = a.output / (('profile-' if profile else 'unprofiled-')
                                       + split + '-' + prefix + '-c' + str(c))
                    wrapper = a.output / (case.name + '-serve.sh')
                    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                        'test "${1:?}" = executorch\n'
                        + ('export SCALING_PROFILE_CASE=' + shlex.quote(str(case)) + '\n' if profile else '')
                        + 'exec ' + shlex.join([str(a.python), '-B', str(Path(__file__).resolve()),
                            '--phase', 'profile' if profile else 'serve', *common]) + '\n')
                    argv = [str(a.benchmark_python), '-B', str(pkg / 'run_case_24h.py'),
                        '--wrapper', str(wrapper), '--backend', 'executorch', '--model',
                        'gemma4-12b-it', '--variant', 'unchanged-existing2048-scaling-diagnostic',
                        '--prefix-cache', prefix, '--client', str(a.client), '--prompts',
                        str(a.inputs / (split + '_benchmark_prompts.json')), '--correctness-root',
                        str(a.inputs / 'correctness'), '--python', str(a.benchmark_python),
                        '--concurrency', str(c), '--port', str(a.port), '--output', str(case),
                        '--lock', str(a.lock), '--mode', 'performance', '--split', split,
                        '--performance-rounds', '1']
                    state('collecting-profile' if profile else 'collecting-unprofiled-control', case=str(case))
                    (a.output / (case.name + '-command.json')).write_text(
                        json.dumps(dict(argv=argv, profile=profile), indent=2) + '\n')
                    with (a.output / (case.name + '-controller.log')).open('x') as f:
                        subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=2400)
                    if profile:
                        report = case / 'graph-trace.nsys-rep'
                        assert report.is_file() and report.stat().st_size > 0, 'No usable trace; no idle inference'
                        with (case / 'nsys-export.log').open('x') as f:
                            subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output',
                                str(case / 'graph-trace.sqlite'), str(report)], stdout=f,
                                stderr=subprocess.STDOUT, check=True, timeout=300)
                    completed.append(case.name)
        state('collected-pending-fixed-output-and-causal-review',
              fixed_output_control_pending=True, profiling_timings_are_not_main_data=True)
    except Exception as e:
        (a.output / 'exception.txt').write_text(traceback.format_exc())
        state('failed-preserve-evidence', error_type=type(e).__name__, detail=str(e)[-2000:])
        raise


if __name__ == '__main__':
    main()
