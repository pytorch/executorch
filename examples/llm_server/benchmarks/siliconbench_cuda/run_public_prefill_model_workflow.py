"""Generate prefill evidence locally, then export an isolated MG mini pair.

Inputs are public GGUF and reviewed source. No remote validation metadata is
required. Never restart a screening job because an observation timed out.
All GPU stages use the existing task lock and preserve failed work.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while raw := f.read(8 * 1024 * 1024):
            h.update(raw)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--screen-job', type=Path, required=True)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--port-kind', choices=['small-bulk','prepared-metadata','native-branch-control'], default='small-bulk')
    p.add_argument('--reuse-original-workflow', type=Path,
                   help='Reuse verified reference artifact and delegate proof; export only the routed public arm')
    a = p.parse_args()
    assert not a.job_dir.exists()
    a.job_dir.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'public-source-manifest.json').read_text())
    assert manifest['public_only_source']
    for name in ['benchmark_public_grouped_w4a16.py',
                 'probe_public_prefill_ET_delegate.py',
                 'export_public_prefill_MG_mini.py',
                 'public_prefill_export_hook.py', 'public_bulk_w4a16_aoti.py',
                 'public_grouped_w4a16.py', 'public_grouped_w4a16_aoti.py']:
        assert sha(package / name) == manifest['files'][name]['sha256']
    if a.port_kind in ['prepared-metadata','native-branch-control']:
        for name in ['public_prefill_prepared_groups.py','public_prefill_prepared_groups_aoti.py']:
            assert sha(package/name) == manifest['files'][name]['sha256']
    if a.port_kind == 'native-branch-control':
        name = 'public_prefill_native_branch_control.py'
        assert sha(package/name) == manifest['files'][name]['sha256']

    def state(status, **fields):
        record = dict(status=status, controller_pid=os.getpid(),
                      UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      model_accepted=False, HTTP_accepted=False, new_KA=False,
                      external_metadata_copied=False, **fields)
        pending = a.job_dir / 'progress.pending.json'
        pending.write_text(json.dumps(record, indent=2) + '\n')
        pending.replace(a.job_dir / 'progress.json')

    state('waiting-existing-local-screen')
    deadline = time.monotonic() + 7200
    try:
        while True:
            progress = a.screen_job / 'progress.json'
            assert progress.exists(), 'Existing local screen must have a live handle'
            s = json.loads(progress.read_text())
            if s['status'] == 'completed':
                break
            assert s['status'] != 'failed', s
            # A state file is insufficient: confirm the actual existing owner.
            os.kill(int(s['controller_pid']), 0)
            assert time.monotonic() < deadline, 'Observation expired; do not restart screen'
            time.sleep(15)
        screen = json.loads((a.screen_job / 'results.json').read_text())
        assert screen['GGUF_SHA256'] == sha(a.gguf)
        assert set(screen['weight_sources']) >= {'ffn_gate_up', 'ffn_down'}
        assert {r['entry'] for r in screen['results']} >= {
            'bulk72', 'bulk128', 'bulk512', 'bulk2048'}

        def run(command, label, timeout):
            state('running-' + label, command=[str(v) for v in command])
            with (a.job_dir / (label + '.log')).open('x') as log:
                child = subprocess.Popen([str(v) for v in command],
                                         stdout=log, stderr=subprocess.STDOUT)
                state('running-' + label, child_pid=child.pid,
                      command=[str(v) for v in command])
                try:
                    assert child.wait(timeout=timeout) == 0, label
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise

        if a.reuse_original_workflow:
            reference = json.loads((a.reuse_original_workflow / 'mini-pair-exports.json').read_text())
            delegate = a.reuse_original_workflow / 'linear-delegate'
            run([sys.executable, '-B', package / 'probe_public_prefill_CPU_fake_trace.py',
                 '--et-source', a.et_source, '--output', a.job_dir / 'CPU-fake-route-proof.json',
                 '--port-kind',a.port_kind],
                'CPU-fake-route', 300)
        else:
            delegate = a.job_dir / 'linear-delegate'
            run([sys.executable, '-B', package / 'probe_public_prefill_ET_delegate.py',
                 '--et-source', a.et_source, '--gguf', a.gguf,
                 '--output', delegate, '--screen-results', a.screen_job / 'results.json',
                 '--lock', a.lock], 'linear-delegate', 7200)
        if a.port_kind in ['prepared-metadata','native-branch-control']:
            # A prior small-bulk delegate does not prove this new operator.
            delegate = a.job_dir / 'prepared-linear-delegate'
            run([sys.executable,'-B',package/'probe_public_prefill_ET_delegate.py',
                 '--et-source',a.et_source,'--gguf',a.gguf,'--output',delegate,
                 '--screen-results',a.screen_job/'results.json','--lock',a.lock,
                 '--native-branch-control' if a.port_kind == 'native-branch-control' else '--prepared-groups'],'prepared-linear-delegate',7200)
        proof = json.loads((delegate / 'results.json').read_text())
        assert proof['ET_edge_and_CUDA_serialization_pass'] and len(proof['artifacts']) == 4
        artifacts = {}
        for arm in ['original', 'public']:
            if arm == 'original' and a.reuse_original_workflow:
                output = Path(reference['artifacts'][arm]['path'])
                job = a.reuse_original_workflow / 'mini-original-export'
            else:
                output = a.job_dir / ('mini-' + arm + '-artifact')
                job = a.job_dir / ('mini-' + arm + '-export')
                command = [sys.executable, '-B', package / 'export_public_prefill_MG_mini.py',
                 '--et-source', a.et_source, '--gguf', a.gguf,
                 '--delegate-proof', delegate / 'results.json', '--arm', arm,
                 '--output', output, '--job', job, '--lock', a.lock,
                 '--port-kind',a.port_kind]
                if a.reuse_original_workflow:
                    command += ['--seed-reference-export-cache',
                                a.reuse_original_workflow / 'mini-original-export' / 'cache']
                run(command, 'mini-' + arm + '-export', 7500)
            assert json.loads((job / 'progress.json').read_text())['status'] == 'completed'
            audit = json.loads((output / 'public-prefill-mini-artifact-audit.json').read_text())
            assert audit['n_layers'] == 4 and audit['GGUF_SHA256'] == screen['GGUF_SHA256']
            assert audit['static_forwards'] == [1, 2, 4, 8, 16, 32, 64]
            for name, record in audit['artifacts'].items():
                assert sha(output / name) == record['SHA256']
            artifacts[arm] = dict(path=str(output), audit=audit)
        (a.job_dir / 'mini-pair-exports.json').write_text(json.dumps(dict(
            artifacts=artifacts, screen_results_SHA256=sha(a.screen_job / 'results.json'),
            delegate_results_SHA256=sha(delegate / 'results.json'),
            mini_runtime_numeric_and_perf_pending=True,
            full_model_F1_and_HTTP_pending=True, model_promoted=False), indent=2) + '\n')
        state('mini-pair-exported-pending-runtime-validation')
    except Exception as e:
        state('failed', error_type=type(e).__name__, detail=str(e)[-2000:])
        raise


if __name__ == '__main__':
    main()
