"""Qualify an allocator SDK using preserved MG mini graphs and real HTTP.

Wait for the existing native build, generate matching old/new probe code,
collect four fresh rounds, and reconstruct raw storage/performance evidence.
Keep every prior artifact and gate; this script never deploys a full model.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-job', 'original-mini-preparation', 'tokenizer',
                 'lock', 'nsys', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **extra):
        f = a.output / 'progress.pending.json'
        f.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False, main_candidate_accepted=False,
            full_model_export_started=False, **extra), indent=2) + '\n')
        f.replace(a.output / 'progress.json')

    def run(label, script, argv):
        cmd = [sys.executable, '-B', '-u', str(package / script), *map(str, argv)]
        (a.output / (label + '-command.json')).write_text(
            json.dumps(dict(argv=cmd), indent=2) + '\n')
        with (a.output / (label + '.log')).open('x') as log:
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, timeout=14400)

    try:
        until = time.monotonic() + 90 * 60
        while True:
            d = json.loads((a.native_job / 'progress.json').read_text())
            assert d['status'] != 'failed', d
            alias = a.native_job / 'same-artifact-mini-preparation'
            if d['status'] == 'native-GPU-tests-passed' and (alias / 'progress.json').is_file():
                break
            handle = Path('/proc') / str(d['controller_pid']) / 'cmdline'
            actual = handle.read_bytes() if handle.is_file() else b''
            if not actual:
                refreshed = json.loads((a.native_job / 'progress.json').read_text())
                if refreshed['status'] == 'native-GPU-tests-passed' and (alias / 'progress.json').is_file():
                    break
            assert str(a.native_job).encode() in actual
            assert b'prepare_independent_session_coalesced_growth_trim.py' in actual
            assert time.monotonic() < until, 'Preserve verified-live build after observation timeout'
            state('waiting-confirmed-live-allocator-native', native_status=d['status'],
                  native_PID=d['controller_pid'])
            time.sleep(20)
        gate = verify_runtime(a.native_job, package)
        original = json.loads((a.original_mini_preparation / 'native-gate.json').read_text())
        assert original['native_GPU_tests_pass'] and original['test_count'] == 74
        assert not original.get('actual_native_source_change')
        assert original['source_spec_SHA256'] == gate['source_spec_SHA256']
        assert original['unchanged_controls'] == gate['unchanged_controls']
        for name, expected in original['binaries'].items():
            assert sha(name) == expected, name
        state('building-original-matched-mini-probe')
        run('original-probe', 'prepare_rewind_kv_probe.py', [
            '--preparation-job', a.original_mini_preparation, '--python', sys.executable,
            '--output', a.output / 'original-rewind-probe'])
        state('collecting-original-matched-mini-latency')
        run('original-latency', 'run_independent_session_mini_model_latency.py', [
            '--preparation-job', a.original_mini_preparation, '--mini-profile', 'muse-glimmer-first4',
            '--rewind-probe-job', a.output / 'original-rewind-probe',
            '--native-sampler', 'off', '--lock', a.lock, '--nsys', a.nsys,
            '--output', a.output / 'original-latency'])
        state('preparing-isolated-current-native-HTTP-dependencies')
        run('HTTP-dependencies', 'prepare_mg_http_python_dependencies.py', [
            '--native-job', a.native_job, '--python', sys.executable,
            '--output', a.output / 'HTTP-dependencies'])
        state('collecting-mini-native-and-HTTP-controls')
        run('mini-controls', 'run_growth_allocator_mg_mini_controls.py', [
            '--native-job', a.native_job, '--tokenizer', a.tokenizer,
            '--http-python-dependencies-job', a.output / 'HTTP-dependencies',
            '--lock', a.lock, '--nsys', a.nsys, '--port', a.port,
            '--output', a.output / 'mini-controls'])
        state('independently-reviewing-mini-raw-evidence')
        run('qualification', 'qualify_independent_session_mini_for_full.py', [
            '--mini-job', a.output / 'mini-controls', '--native-job', a.native_job,
            '--previous-latency-job', a.output / 'original-latency',
            '--profile', 'muse-glimmer-first4', '--output', a.output / 'mini-reviews'])
        qpath = a.output / 'mini-reviews/mini-qualification.json'
        q = json.loads(qpath.read_text())
        assert q['runtime_native_gate_SHA256'] == sha(a.native_job / 'native-gate.json')
        assert q['mini_storage_correctness_pass'] and q['mini_HTTP_memory_pass']
        verify_runtime(a.native_job, package)
        result = dict(mini_qualification=str(qpath.resolve()),
            mini_qualification_SHA256=sha(qpath),
            mini_cold_and_steady_performance_pass=q['mini_cold_and_steady_performance_pass'],
            positive_performance_signals=q['positive_performance_signals'],
            full_model_export_for_qualification_allowed=q['full_model_export_for_qualification_allowed'],
            full_model_export_started=False, full_model_qualified=False,
            main_candidate_accepted=False)
        (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        state('allocator-mini-qualified-pending-full-model' if q['full_model_export_for_qualification_allowed']
              else 'mini-performance-gate-not-passed', **{k: result[k] for k in (
                  'mini_qualification_SHA256', 'positive_performance_signals')})
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
