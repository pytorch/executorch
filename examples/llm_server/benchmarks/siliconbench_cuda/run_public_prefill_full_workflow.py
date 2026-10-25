"""Qualify a prepared-metadata prefill candidate against complete primary MG.

Wait for the existing mini trial; never restart it. Preserve the accepted
primary artifact, cache, native servers and records. Export the public arm
with the primary 52-layer/21-owner geometry, then check executor numerics.
An explicit option preserves a failed auxiliary RMS screen while collecting
the user's full-model F1 assessment. All records remain provisional until an
independent review selects an arm; no gate is silently changed to passing.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mini-runtime-job', type=Path, required=True)
    p.add_argument('--mini-workflow-job', type=Path, required=True)
    p.add_argument('--reference-artifact', type=Path, required=True)
    p.add_argument('--reference-export-job', type=Path, required=True)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--CPU-build-progress', type=Path, required=True)
    p.add_argument('--accepted-main-review', type=Path, required=True)
    p.add_argument('--base-wrapper', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--correctness-root', type=Path, required=True)
    p.add_argument('--benchmark-python', required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=19110)
    p.add_argument('--quality-max-tokens', type=int, default=1024)
    p.add_argument('--quality-chat-template-kwargs', type=json.loads,
                   default=dict(enable_thinking=False, reasoning_strength='low'))
    p.add_argument('--collect-F1-after-numerical-screen-failure', action='store_true',
                   help='Keep failed auxiliary RMS gates visible and collect full-model F1; no automatic promotion')
    a = p.parse_args()
    assert not a.job_dir.exists(), 'Preserve every earlier attempt'
    a.job_dir.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **fields):
        pending = a.job_dir / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model=True, model_promoted=False, new_KA=False, **fields), indent=2)+'\n')
        pending.replace(a.job_dir / 'progress.json')

    def run(command, label, timeout):
        with (a.job_dir / (label+'.log')).open('x') as log:
            child = subprocess.Popen([str(v) for v in command], stdout=log, stderr=subprocess.STDOUT)
            state('running-'+label, child_pid=child.pid)
            try:
                assert child.wait(timeout=timeout) == 0, label
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=120)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                raise

    try:
        state('waiting-existing-mini-validation')
        deadline = time.monotonic()+20000
        mini_path = a.mini_runtime_job / ('numerical-preflight.json' if a.collect_F1_after_numerical_screen_failure else 'results.json')
        while not mini_path.exists():
            s = json.loads((a.mini_runtime_job / 'progress.json').read_text())
            assert s['status'] != 'failed', s
            os.kill(s['controller_pid'], 0)
            assert time.monotonic() < deadline, 'Observation expired; do not restart mini trial'
            time.sleep(15)
        mini = json.loads(mini_path.read_text())
        if not a.collect_F1_after_numerical_screen_failure:
            assert mini['mini_numeric_pass'] and mini['four_rounds_first_discarded']
            assert all(r['numerical_pass'] for r in mini['numerical'])
        else:
            assert mini['numerical_gate'] == .005
            assert len(mini['numerical']) == 9 and all(r['argmax_identical'] for r in mini['numerical'])
            assert len(mini['raw']) == 2 and all(r['round'] == 1 for r in mini['raw'])
        (a.job_dir/'retained-mini-numerical-screen.json').write_text(json.dumps(dict(
            source=str(mini_path), SHA256=sha(mini_path),
            numerical_gate=.005, numerical=mini['numerical'],
            original_numerical_pass=all(r['numerical_pass'] for r in mini['numerical']),
            separate_complete_F1_qualification=a.collect_F1_after_numerical_screen_failure,
            model_promoted=False),indent=2)+'\n')
        delegate = a.mini_workflow_job / 'prepared-linear-delegate/results.json'
        proof = json.loads(delegate.read_text())
        assert proof['port_kind'] == 'prepared-metadata' and proof['ET_edge_and_CUDA_serialization_pass']
        reference = json.loads((a.reference_artifact / 'artifact-audit.json').read_text())
        assert reference['constant_metadata']['get_n_caches'] == 52
        assert reference['max_resident_sequences'] == 21 and reference['physical_columns'] == 22
        assert reference['constant_metadata']['get_max_prefill_chunk'] == 2048
        assert reference['context'] == 131072 and reference['selected_output_bound'] == [64, 202048]
        assert reference['static_methods'] == [1,2,4,8,16,32,64]
        for name, item in reference['artifacts'].items():
            assert sha(a.reference_artifact/name) == item['sha256']
        assert json.loads((a.reference_export_job/'progress.json').read_text())['status'] == 'completed'
        cache = a.reference_export_job/'cache'
        cache_bytes = sum(f.stat().st_size for name in ['inductor','triton']
                          for f in (cache/name).rglob('*') if f.is_file())
        # Weight materialization and final serialization temporarily coexist.
        # Reserve three full blobs plus cache and headroom, not only the final PTD.
        required = 3*reference['artifacts']['aoti_cuda_blob.ptd']['bytes']+cache_bytes+4*1024**3
        available = shutil.disk_usage(a.job_dir).free
        (a.job_dir/'export-storage-preflight.json').write_text(json.dumps(dict(
            available_bytes=available, required_bytes=required, temporary_full_blob_copies_budgeted=3,
            reference_cache_bytes=cache_bytes, original_reference_modified=False),indent=2)+'\n')
        assert available > required, 'Insufficient export staging storage; retain existing files'
        output = a.job_dir/'public-full-artifact'
        export = a.job_dir/'public-full-export'
        run([sys.executable,'-B',package/'export_public_prefill_MG_mini.py',
            '--et-source',a.et_source,'--gguf',a.gguf,'--delegate-proof',delegate,
            '--arm','public','--port-kind','prepared-metadata','--n-layers','52',
            '--max-kv-seqs','21','--reference-full-artifact',a.reference_artifact,
            '--seed-reference-export-cache',cache,'--output',output,'--job',export,
            '--lock',a.lock], 'full-model-export', 22800)
        public = json.loads((output/'public-prefill-mini-artifact-audit.json').read_text())
        assert public['n_layers'] == 52 and public['max_kv_seqs'] == 21
        assert public['primary_full_reference_verified']
        normalized = dict(n_layers=52,logical_context=131072,max_step_tokens=2048,max_kv_seqs=21,
            static_forwards=reference['static_methods'],selected_logits=reference['selected_output_bound'],
            GGUF_SHA256=reference['GGUF_sha256'],
            artifacts={name:dict(SHA256=item['sha256'],bytes=item['bytes'])
                       for name,item in reference['artifacts'].items()})
        pair = dict(artifacts=dict(original=dict(path=str(a.reference_artifact),audit=normalized),
                                  public=dict(path=str(output),audit=public)),
            full_model_F1_and_HTTP_pending=True, model_promoted=False)
        (a.job_dir/'model-pair-exports.json').write_text(json.dumps(pair,indent=2)+'\n')
        runtime = a.job_dir/'full-runtime'
        runtime_command = [sys.executable,'-B',package/'run_public_prefill_mini_runtime.py',
            '--workflow-job',a.job_dir,'--workflow-pair-filename','model-pair-exports.json',
            '--expected-n-layers','52','--runtime-sessions','16',
            '--CPU-build-progress',a.CPU_build_progress,'--et-source',a.et_source,
            '--accepted-main-review',a.accepted_main_review,'--job-dir',runtime,
            '--lock',a.lock,'--warmup','5','--repetitions','12','--numerical-only']
        if a.collect_F1_after_numerical_screen_failure:
            runtime_command += ['--continue-after-numerical-screen-failure']
        run(runtime_command, 'full-model-executor', 21600)
        measured = json.loads((runtime/'results.json').read_text())
        assert measured['n_layers'] == 52 and measured['numerical_only']
        if not a.collect_F1_after_numerical_screen_failure:
            assert measured['mini_numeric_pass'] and all(r['numerical_pass'] for r in measured['numerical'])
        (a.job_dir/'retained-full-numerical-screen.json').write_text(json.dumps(dict(
            source=str(runtime/'results.json'), SHA256=sha(runtime/'results.json'),
            numerical_gate=.005, numerical=measured['numerical'],
            original_numerical_pass=measured['mini_numeric_pass'],
            separate_complete_F1_qualification=a.collect_F1_after_numerical_screen_failure,
            model_promoted=False),indent=2)+'\n')
        # Use the unchanged primary server interface. Only its artifact input
        # differs, and every new result lives below this isolated job directory.
        assert sha(a.client) == '3a748babe2fc62d9f10c08892de65b57954d9661bd0689988ac3d6d5cbabf982'
        assert sha(a.base_wrapper) == '4f115fdea2368ee7ea075b27542d63bf3d7ae61b676751541393d2ff323d935a'
        os.environ.update(SB_MODEL='muse-glimmer', SB_ET_CACHEOFF_ARTIFACT=str(output),
                          SB_ET_PREFIX_ENTRIES='4',SB_CHECKPOINT='0')
        matrix_command = [a.benchmark_python,'-B',package/'run_matrix_24h.py',
            '--wrapper',a.base_wrapper,'--model','muse-glimmer-30b','--client',a.client,
            '--prompts',a.prompts,'--correctness-root',a.correctness_root,
            '--python',a.benchmark_python,
            '--lock',a.lock,'--port',a.port,'--variants','executorch',
            '--cache-modes','off','on','--concurrencies','16','8','4','2','1']
        assert a.quality_max_tokens > 0 and isinstance(a.quality_chat_template_kwargs,dict)
        run(matrix_command+['--output',a.job_dir/'server-performance','--phases','performance'],
            'full-model-server-performance', 21600)
        run(matrix_command+['--output',a.job_dir/'server-correctness','--phases','correctness',
            '--correctness-max-tokens',a.quality_max_tokens,
            '--correctness-chat-template-kwargs',json.dumps(a.quality_chat_template_kwargs)],
            'full-model-server-F1', 57600)
        state('completed-pending-independent-full-model-selection',
              executor_numeric_pass=measured['mini_numeric_pass'],
              server_perf_memory_and_F1_collected=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-2000:])
        raise


if __name__ == '__main__':
    main()
