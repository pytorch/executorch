"""Resume an inspected, terminal matrix without repeating completed cells.

Retry only explicitly reviewed failed cells. Preserve their complete directories
and controller logs, and rerun their full protocol with unchanged source inputs.
This controller changes no framework, server option, model or client workload.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--previous-controller', type=int, required=True)
    p.add_argument('--retry-case', action='append', default=[])
    p.add_argument('--wrapper', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--native-inputs', type=Path)
    p.add_argument('--correctness-root', type=Path, required=True)
    p.add_argument('--case-driver', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--port', type=int, default=19082)
    p.add_argument('--cpp-off-build', default='build-stock-native-evaluation')
    args = p.parse_args()
    matrix = args.matrix
    continuation_lock = (matrix/'reviewed-continuation.lock').open('a')
    fcntl.flock(continuation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads((matrix/'progress.json').read_text())
    plan = json.loads((matrix/'matrix.json').read_text())

    def live(pid):
        path = Path('/proc')/str(pid)/'stat'
        return path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] not in ['Z', 'X']

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert old['status'] == 'failed' and old['controller_pid'] == args.previous_controller
    assert not live(args.previous_controller), 'Never resume an active matrix'
    assert all(phase in ['performance', 'correctness'] and cache in ['off', 'on'] and
               c in [1,2,4,8,16] and variant in ['llamacpp-off', 'llamacpp-on', 'vllm', 'sglang', 'executorch']
               for phase, cache, c, variant in plan['cases'])
    names = [f'{phase}-agent-{variant}-prefix-{cache}-c{c}' for phase, cache, c, variant in plan['cases']]
    assert len(names) == len(set(names))
    assert set(args.retry_case) <= set(names)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    hashes = dict(wrapper_sha256=sha(args.wrapper), client_sha256=sha(args.client), prompts_sha256=sha(args.prompts))
    driver_hash = sha(args.case_driver)
    completed = []
    retry = []
    for name, (phase, cache, c, variant) in zip(names, plan['cases']):
        cell = matrix/name
        if not cell.exists():
            continue
        progress = json.loads((cell/'progress.json').read_text())
        assert not live(progress['controller_pid']), 'Preserve active case controllers'
        settings = json.loads((cell/'settings.json').read_text())
        assert all(settings[key] == value for key, value in hashes.items())
        assert settings['model'] == plan['model'] and settings['variant'] == variant
        assert settings['mode'] == phase and settings['prefix_cache'] == (cache == 'on') and settings['concurrency'] == c
        if progress['status'] == 'completed':
            if phase == 'performance':
                assert settings['performance_rounds'] == 4 and settings['averaged'] == [2,3,4]
                # Initial cells predate explicit round tags. Their recorded
                # memory samples are retained and independently audited by the
                # collector using the four metric completion timestamps.
                assert (cell/'memory-summary.json').is_file() and (cell/'memory-samples.json').is_file()
                for n in [1,2,3,4]:
                    measured = json.loads((cell/f'repeat-{n}/metrics.json').read_text())['concurrency_results'][0]
                    assert measured['successful'] == 100 and measured['failed'] == 0
            else:
                assert (cell/'correctness-aggregate.json').is_file()
            completed.append(name)
        else:
            assert name in args.retry_case and progress['status'] == 'failed', 'Review each unsuccessful case explicitly'
            retry.append(name)
    assert set(old['completed']) == set(completed), 'Completed cases differ from the inspected matrix state'
    assert set(retry) == set(args.retry_case)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = matrix.parent/'preserved-matrix-failures'/stamp
    archive.mkdir(parents=True)
    claim = archive/'continuation-claim.json'
    with claim.open('x') as output:
        json.dump(dict(controller_pid=os.getpid(), previous_controller=args.previous_controller,
                       cases_preserved=retry, cases_skipped=completed, input_hashes=hashes,
                       case_driver_sha256=driver_hash, unchanged_protocol=True), output, indent=2)
    (matrix/'progress.json').rename(archive/'original-matrix-progress.json')
    for name in retry:
        (matrix/name).rename(archive/name)
        log = matrix/(name+'.log')
        if log.exists():
            log.rename(archive/(name+'.log'))

    def state(status, **fields):
        path = matrix/'progress.pending.json'
        path.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                         utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         completed=completed, reviewed_retry_archive=str(archive), **fields), indent=2)+'\n')
        path.replace(matrix/'progress.json')

    try:
        for name, (phase, cache, c, variant) in zip(names, plan['cases']):
            if name in completed:
                continue
            assert sha(args.case_driver) == driver_hash and all(sha(getattr(args, key.removesuffix('_sha256'))) == value for key, value in hashes.items())
            state('running', current_case=name, total_cases=len(names))
            command = [args.python, '-B', str(args.case_driver), '--mode', phase,
                       '--quality-scope', 'full' if c in [1,16] else 'smoke',
                       '--wrapper', str(args.wrapper), '--backend', 'llamacpp' if variant.startswith('llamacpp-') else variant,
                       '--variant', variant, '--model', plan['model'], '--prefix-cache', cache,
                       '--client', str(args.client), '--prompts', str(args.prompts),
                       '--correctness-root', str(args.correctness_root), '--python', args.python,
                       '--concurrency', str(c), '--port', str(args.port), '--output', str(matrix/name),
                       '--lock', str(args.lock), '--cpp-build', args.cpp_off_build if variant == 'llamacpp-off' else 'build-stock-cublas']
            if args.native_inputs:
                command += ['--native-inputs', str(args.native_inputs)]
            with (matrix/(name+'.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads((matrix/name/'progress.json').read_text())['status'] == 'completed'
            completed.append(name)
        state('completed', total_cases=len(names), final_quality_cache_memory_and_best_ET_acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
