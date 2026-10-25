"""Qualify an existing near-cap worker after its two-cell diagnostic.

Reuse the completed OFF/c1/c16 performance cells. Run only the remaining
eight standard performance cells and the ten agreed correctness cells.
Model, wrapper, weights, native runtimes and earlier results are unchanged.
This collects evidence and never automatically selects a model.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-job', 'client', 'prompts', 'correctness-root', 'lock', 'job-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert not args.job_dir.exists(), 'Keep every earlier attempt'
    args.job_dir.mkdir(parents=True)

    def state(status, **fields):
        pending = args.job_dir / 'progress.pending.json'
        pending.write_text(json.dumps(dict(
            status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_promoted=False, new_KA=False, primary_results_changed=False,
            **fields), indent=2) + '\n')
        pending.replace(args.job_dir / 'progress.json')

    try:
        state('waiting-existing-two-cell-server-diagnostic')
        deadline = time.monotonic() + 86400
        while True:
            parent = json.loads((args.parent_job / 'progress.json').read_text())
            if parent['status'] == 'completed-pending-independent-probe-and-server-review':
                break
            assert parent['status'] != 'failed', 'Retain the failed parent; do not rerun it'
            os.kill(parent['controller_pid'], 0)
            assert time.monotonic() < deadline, 'Observation expired; do not restart work'
            time.sleep(30)
        protocol = json.loads((args.parent_job / 'server-protocol.json').read_text())
        wrapper = Path(protocol['wrapper'])
        for path, expected in [(wrapper, protocol['wrapper_SHA256']),
                               (args.client, protocol['client_SHA256']),
                               (args.prompts, protocol['prompts_SHA256'])]:
            assert digest(path) == expected, str(path)
        assert protocol['logical_context'] == 131072 and protocol['initial_KV'] == 512
        assert protocol['active_sessions'] == protocol['decode_limit'] == 16
        assert protocol['bulk_A16'] and protocol['static_graphs']
        reused = []
        for c in [1, 16]:
            path = args.parent_job / 'server-diagnostic' / (
                f'performance-agent-executorch-prefix-off-c{c}')
            case = json.loads((path / 'progress.json').read_text())
            assert case['status'] == 'completed'
            aggregate = json.loads((path / 'performance-aggregate.json').read_text())
            for repeat in [1, 2, 3, 4]:
                metric = json.loads((path / f'repeat-{repeat}/metrics.json').read_text())[
                    'concurrency_results'][0]
                assert metric['successful'] == 100 and metric['failed'] == 0
            reused.append(dict(prefix=False, c=c, path=str(path),
                               aggregate_SHA256=digest(path / 'performance-aggregate.json'),
                               performance=aggregate))
        (args.job_dir / 'reused-completed-performance.json').write_text(
            json.dumps(reused, indent=2) + '\n')
        (args.job_dir / 'server-protocol.json').write_text(
            json.dumps(protocol, indent=2) + '\n')
        model = 'gemma4-12b-it'
        standard = [(phase, cache, c, 'executorch')
                    for phase in ['performance', 'correctness']
                    for cache in ['off', 'on'] for c in [16, 8, 4, 2, 1]]
        perf = [row for row in standard if row[0] == 'performance'
                and (row[1], row[2]) not in [('off', 1), ('off', 16)]]
        assert len(perf) == 8
        plan = args.job_dir / 'remaining-performance-plan.json'
        plan.write_text(json.dumps(dict(model=model,
            purpose='disjoint performance continuation', cases=perf), indent=2) + '\n')
        for phase in ['performance', 'correctness']:
            output = args.job_dir / ('server-' + phase)
            command = [args.python, '-B', str(Path(__file__).with_name('run_matrix_24h.py')),
                '--wrapper', str(wrapper), '--model', model,
                '--client', str(args.client), '--prompts', str(args.prompts),
                '--correctness-root', str(args.correctness_root), '--python', args.python,
                '--output', str(output), '--lock', str(args.lock), '--port', str(args.port),
                '--variants', 'executorch', '--cache-modes', 'off', 'on',
                '--concurrencies', '16', '8', '4', '2', '1', '--phases', phase]
            if phase == 'performance':
                command += ['--case-plan', str(plan)]
            with (args.job_dir / (phase + '-controller.log')).open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state('running-standard-server-' + phase, child_pid=child.pid,
                      reused_performance_cells=2, current_matrix=str(output))
                returncode = child.wait()
                assert returncode == 0, 'Own matrix failed; retain every log and result'
            matrix = json.loads((output / 'progress.json').read_text())
            assert matrix['status'] == 'completed'
            assert len(matrix['completed']) == (8 if phase == 'performance' else 10)
        state('completed-pending-independent-best-policy-selection',
              performance_cells=10, newly_measured_performance_cells=8,
              reused_performance_cells=2, correctness_cells=10)
    except BaseException as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1800:])
        raise


if __name__ == '__main__':
    main()
