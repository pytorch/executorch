"""Collect full52 MG HTTP memory after actual same-PTE native qualification.

Keep mini/full/export/runtime provenance separate. The independent full model
is collected alone without inventing a global comparison or erasing a timing
signal. Task F1 and main performance remain separate gates.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-lifecycle-job', 'native-job', 'mini-qualification',
                 'memory-worker-proof', 'http-python-dependencies-job',
                 'tokenizer', 'lock', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **extra):
        f = a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False, main_candidate_accepted=False, **extra), indent=2)+'\n')
        f.replace(a.output/'progress.json')

    def run(label, script, argv):
        command = [sys.executable, '-B', '-u', str(package/script), *map(str, argv)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command), indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, timeout=21600)

    try:
        until = time.monotonic()+6*3600
        while True:
            d = json.loads((a.native_lifecycle_job/'progress.json').read_text())
            if d['status'] == 'full-MG-single-artifact-native-controls-collected':
                break
            assert d['status'] != 'failed', d
            handle = Path('/proc')/str(d['controller_pid'])/'cmdline'
            command = handle.read_bytes() if handle.is_file() else b''
            if not command:
                refreshed = json.loads((a.native_lifecycle_job/'progress.json').read_text())
                if refreshed['status'] == 'full-MG-single-artifact-native-controls-collected':
                    break
            assert b'run_single_full_MG_lifecycle_after_export.py' in command
            assert str(a.native_lifecycle_job).encode() in command
            assert time.monotonic() < until, 'Preserve verified-live native work after observation timeout'
            state('waiting-confirmed-live-full-MG-native', native_PID=d['controller_pid'],
                  native_status=d['status'])
            time.sleep(20)
        verify_runtime(a.native_job, package)
        native = json.loads((a.native_lifecycle_job/'result.json').read_text())
        assert native['full_profile'] == 'muse-glimmer-52layers'
        assert native['full_independent_storage_and_observed_numerics_pass']
        assert native['actual_runtime_native_gate_SHA256'] == sha(a.native_job/'native-gate.json')
        for name, expected in native['raw_reviews_SHA256'].items():
            assert sha(a.native_lifecycle_job/name) == expected, name
        state('collecting-full52-HTTP-memory', positive_performance_signals=native['positive_performance_signals'])
        run('HTTP-memory', 'run_independent_session_mini_http.py', [
            '--preparation-job', a.native_lifecycle_job/'after-full-artifact',
            '--python', sys.executable, '--tokenizer', a.tokenizer, '--lock', a.lock,
            '--output', a.output/'HTTP-memory', '--port', a.port,
            '--mini-profile', 'muse-glimmer-first4', '--full-model-profile', 'muse-glimmer-52layers',
            '--arm', 'independent', '--full-native-job', a.native_job,
            '--full-MG-native-lifecycle-job', a.native_lifecycle_job,
            '--full-MG-mini-qualification', a.mini_qualification,
            '--memory-worker-proof', a.memory_worker_proof,
            '--http-python-dependencies-job', a.http_python_dependencies_job])
        run('HTTP-review', 'review_independent_session_http_lifecycle.py', [
            '--job', a.output/'HTTP-memory', '--require-native-memory',
            '--output', a.output/'HTTP-independent-review.json'])
        review = json.loads((a.output/'HTTP-independent-review.json').read_text())
        assert review['full_model_profile'] == 'muse-glimmer-52layers'
        assert review['collected_arms'] == ['independent'] and review['matched_requests'] == 0
        assert review['actual_requests'] == 256 and review['server_runs'] == 8
        assert review['completion_lengths_equal_between_arms'] is None
        assert review['native_allocator_memory_verified']
        assert review['automatic_OFF_c16_to_c1_reclaim_verified']
        verify_runtime(a.native_job, package)
        result = dict(full_profile='muse-glimmer-52layers', full_HTTP_memory_pass=True,
            fixed_token_performance_pass=native['fixed_token_same_PTE_runtime_performance_pass'],
            positive_performance_signals=native['positive_performance_signals'],
            HTTP_review_SHA256=sha(a.output/'HTTP-independent-review.json'),
            native_lifecycle_result_SHA256=sha(a.native_lifecycle_job/'result.json'),
            native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),
            HTTP_test_arms=['independent'], global_HTTP_comparison_collected=False,
            all_outputs_reached_limit=review['all_outputs_reached_limit'],
            natural_HTTP_outputs_are_not_task_F1_acceptance=True,
            full_model_qualified=False, main_candidate_accepted=False,
            remaining_gates=['full task F1', 'same-GPU old/new performance', 'main comparison'])
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        state('full-HTTP-memory-collected-pending-F1-and-main',
              positive_performance_signals=result['positive_performance_signals'])
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
