"""Re-review complete MG qualification, then collect the twenty main cases.

Wait only on the actual fresh-quality controller. Every failed or incomplete
attempt stays intact. Main collection requires the strict raw-bound full
qualification writer; this controller never accepts the main results.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from independent_session_full_native_binding import sha


def actual_quality_state(root):
    progress = json.loads((root / 'progress.json').read_text())
    terminal = 'fresh-full-MG-quality-collected-and-recomputed'
    if progress['status'] == terminal:
        return progress, True
    assert progress['status'] != 'failed', progress
    handle = Path('/proc') / str(progress['controller_pid']) / 'cmdline'
    argv = handle.read_bytes().split(b'\0') if handle.is_file() else []
    if not argv:
        progress = json.loads((root / 'progress.json').read_text())
        assert progress['status'] == terminal, 'Quality controller is no longer live'
        return progress, True
    assert any(Path(x.decode()).name == 'run_full_MG_quality_after_matched_native.py'
               for x in argv if x), 'Wrong parent executable'
    assert b'--output' in argv and argv[argv.index(b'--output') + 1] == os.fsencode(root)
    return progress, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('quality-follower-job', 'matched-native-job', 'HTTP-job', 'native-job',
                 'native-lifecycle-job', 'mini-qualification', 'memory-worker-proof',
                 'http-python-dependencies-job', 'tokenizer', 'server-python',
                 'public-inputs', 'lock', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert 1024 <= args.port <= 65535
    assert not args.output.exists(), 'Preserve every prior attempt'
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **extra):
        path = args.output / 'progress.pending.json'
        path.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False, main_candidate_accepted=False, **extra), indent=2) + '\n')
        path.replace(args.output / 'progress.json')

    def run(label, script, tail, timeout):
        argv = [str(args.server_python), '-B', '-u', str(package / script), *map(str, tail)]
        (args.output / (label + '-command.json')).write_text(json.dumps(dict(argv=argv), indent=2) + '\n')
        with (args.output / (label + '.log')).open('x') as log:
            subprocess.run(argv, check=True, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout)

    try:
        until = time.monotonic() + 48 * 3600
        while True:
            progress, complete = actual_quality_state(args.quality_follower_job)
            if complete:
                break
            assert time.monotonic() < until, 'Preserve the verified-live parent after observation timeout'
            state('waiting-confirmed-live-fresh-full-MG-quality',
                  parent_PID=progress['controller_pid'], parent_status=progress['status'])
            time.sleep(20)

        parent_path = args.quality_follower_job / 'result.json'
        parent_hash = sha(parent_path)
        parent = json.loads(parent_path.read_text())
        assert parent['fresh_F1_collected'] and parent['full_task_raw_quality_pass']
        assert parent['full_task_F1_independently_reviewed']
        assert not parent['full_model_qualified'] and not parent['main_candidate_accepted']
        assert parent['native_runtime_gate_SHA256'] == sha(args.native_job / 'native-gate.json')
        assert parent['raw_quality_review_SHA256'] == sha(args.quality_follower_job / 'raw-quality-review.json')

        state('rechecking-complete-full-MG-qualification')
        common = []
        for name in ('native-job', 'native-lifecycle-job', 'HTTP-job', 'mini-qualification',
                     'public-inputs', 'memory-worker-proof', 'http-python-dependencies-job', 'tokenizer'):
            common.extend(['--' + name, getattr(args, name.replace('-', '_'))])
        run('full-qualification', 'write_MG_full_independent_KV_qualification.py', [
            '--package', package, *common, '--matched-native-job', args.matched_native_job,
            '--quality-job', args.quality_follower_job / 'full-quality',
            '--output', args.output / 'full-qualification'], 4 * 3600)
        receipt = args.output / 'full-qualification/full-qualification.json'
        qualified = json.loads(receipt.read_text())
        assert qualified['full_model_qualified'] and not qualified['main_candidate_accepted']
        assert not qualified['unresolved_correctness_or_performance_signals']
        assert sha(parent_path) == parent_hash

        state('collecting-qualified-full-MG-main', full_qualification_receipt_SHA256=sha(receipt))
        run('main-performance', 'run_gemma_independent_full_main.py', [
            '--phase', 'performance', '--full-model-profile', 'muse-glimmer-52layers',
            '--package', package, *common, '--server-python', args.server_python,
            '--lock', args.lock, '--port', args.port,
            '--full-qualification-receipt', receipt,
            '--output', args.output / 'main-performance'], 48 * 3600)
        main_result = args.output / 'main-performance/result.json'
        collected = json.loads(main_result.read_text())
        assert len(collected['completed']) == 20
        assert collected['raw_main_acceptance_pending'] and not collected['main_candidate_accepted']
        assert sha(parent_path) == parent_hash
        (args.output / 'result.json').write_text(json.dumps(dict(
            full_profile='muse-glimmer-52layers', fresh_quality_parent_result_SHA256=parent_hash,
            full_qualification_receipt_SHA256=sha(receipt), main_result_SHA256=sha(main_result),
            main_cases_collected=20, full_model_qualified=True, main_candidate_accepted=False,
            raw_main_review_and_same_GPU_accepted_baseline_comparison_pending=True), indent=2) + '\n')
        state('qualified-MG-main-collected-pending-independent-review')
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
