"""Finish current inference, retire its CPU queue, and preserve protocol records.

This is for replacing a native chat endpoint that rewrites an input. Only the
source-verified CPU matrix parent is held. No server or case is signalled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from enforce_owned_tuning_deadline import identity
from handoff_native_c1_quality import ParentHold, interrupted
from prioritize_native_performance import wait_stopped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--parent-pid', type=int, required=True)
    p.add_argument('--parent-start', type=int, required=True)
    p.add_argument('--parent-command-sha256', required=True)
    p.add_argument('--parent-script-sha256', required=True)
    p.add_argument('--replacement-plan', type=Path, required=True)
    p.add_argument('--native-inputs', type=Path, required=True)
    p.add_argument('--native-inputs-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    assert 1 <= a.timeout <= 7200 and not a.output.exists()
    expected = (a.parent_pid, a.parent_start)
    original = json.loads((a.matrix / 'matrix.json').read_text())
    plan = json.loads(a.replacement_plan.read_text())
    cases = {('performance', cache, c, 'sglang') for cache in ['off', 'on']
             for c in [1, 2, 4, 8, 16]}
    assert original['model'] == plan['model'] == 'muse-glimmer-30b'
    assert {tuple(row) for row in original['cases']} == cases
    assert {tuple(row) for row in plan['cases']} == cases
    assert hashlib.sha256(a.native_inputs.read_bytes()).hexdigest() == a.native_inputs_sha256
    inputs = json.loads(a.native_inputs.read_text())
    assert len(inputs['prompts']) == 100
    assert sum(map(len, inputs['prompts'].values())) == 431510

    def verify_parent():
        assert identity(a.parent_pid) == expected
        raw = Path('/proc', str(a.parent_pid), 'cmdline').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == a.parent_command_sha256
        args = [x.decode() for x in raw.split(b'\0') if x]
        script = next(Path(x) for x in args if x.endswith('/run_matrix_24h.py'))
        assert hashlib.sha256(script.read_bytes()).hexdigest() == a.parent_script_sha256
        assert args[args.index('--output') + 1] == str(a.matrix)
        assert args[args.index('--model') + 1] == 'muse-glimmer-30b'
        assert args[args.index('--variants') + 1] == 'sglang'
        assert args[args.index('--phases') + 1] == 'performance'

    verify_parent()
    a.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def save(status, **fields):
        tmp = a.output / 'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields), indent=2) + '\n')
        tmp.replace(a.output / 'progress.json')

    try:
        with ParentHold(a.parent_pid, expected) as hold:
            wait_stopped(a.parent_pid, expected)
            verify_parent()
            state = json.loads((a.matrix / 'progress.json').read_text())
            assert state['status'] == 'running' and state['controller_pid'] == a.parent_pid
            current = state['current_case']
            assert current.startswith('performance-agent-sglang-prefix-')
            cell = a.matrix / current
            child = json.loads((cell / 'progress.json').read_text())['controller_pid']
            child_id = identity(child)
            if child_id:
                stat = Path('/proc', str(child), 'stat').read_text().rsplit(')', 1)[1].split()
                assert int(stat[1]) == a.parent_pid
            save('waiting-current-inference-with-CPU-parent-held', current_case=current,
                 retained_completed=state['completed'], inference_not_signalled=True)
            end = time.monotonic() + a.timeout
            while True:
                verify_parent()
                s = json.loads((cell / 'progress.json').read_text())
                assert s['status'] not in ['failed', 'cancelled', 'aborted']
                if s['status'] == 'completed' and (cell / 'memory-summary.json').exists() and identity(child) is None:
                    break
                assert time.monotonic() < end, 'Restore CPU parent on timeout'
                time.sleep(5)
            complete = list(state['completed']) + ([current] if current not in state['completed'] else [])
            hashes = {}
            for name in complete:
                q = a.matrix / name
                assert json.loads((q / 'progress.json').read_text())['status'] == 'completed'
                assert (q / 'memory-summary.json').exists()
                for n in range(1, 5):
                    assert len((q / f'repeat-{n}/outputs.jsonl').read_text().splitlines()) == 100
                    m = json.loads((q / f'repeat-{n}/metrics.json').read_text())['concurrency_results'][0]
                    assert m['successful'] == 100 and m['failed'] == 0
                for f in q.rglob('*'):
                    if f.is_file() and f.suffix in ['.json', '.jsonl']:
                        hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            hold.stop_after_child_finished()
        assert identity(a.parent_pid) is None
        for path, sha in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        record = dict(original_CPU_parent_terminal=True, current_inference_completed_without_cancel=True,
                      original_parent_identity=expected, current_case=current, excluded_completed_cases=complete,
                      preserved_response_metric_SHA256=hashes, excluded_from_primary=True,
                      reason='Native chat endpoint rewrites one final assistant tool-call input',
                      all10_replacement_cases_preserved=True, replacement_plan=str(a.replacement_plan),
                      native_inputs=str(a.native_inputs), native_inputs_SHA256=a.native_inputs_sha256)
        marker = a.matrix / 'retirement-native-input-protocol-replacement.json'
        assert not marker.exists()
        marker.write_text(json.dumps(record, indent=2) + '\n')
        save('verified-original-CPU-parent-terminal', **record)
        print('TASK_JSON:' + json.dumps(record), flush=True)
    except BaseException as exc:
        save('failed', error_type=type(exc).__name__, verified_CPU_parent_restored=True)
        raise


if __name__ == '__main__':
    main()
