"""Evaluate an audited MG prefix-capable artifact without promoting it.

Wait for the existing export and preparation services, validate source and
artifact provenance, and run isolated server cases with the shared GPU lock.
One snapshot is an initial feasible policy, not a claim of best performance.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-progress', type=Path, required=True)
    parser.add_argument('--export-program', required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--preparation-output', type=Path, required=True)
    parser.add_argument('--preparation-program', required=True)
    parser.add_argument('--data-revision', required=True)
    parser.add_argument('--base-wrapper', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-entries', type=int, choices=[1,16], default=1)
    parser.add_argument('--checkpoint-interval', type=int, choices=[0,512,1024], default=0)
    parser.add_argument('--cache-modes', nargs='+', choices=['off','on'], default=['off','on'])
    parser.add_argument('--concurrencies', type=int, nargs='+', default=[1,16])
    parser.add_argument('--port', type=int, default=19084)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve all prior artifacts and results'
    assert len(args.data_revision) == 40
    assert set(args.concurrencies) <= {1,2,4,8,16}
    assert len(args.concurrencies) == len(set(args.concurrencies))
    for program in [args.export_program, args.preparation_program]:
        assert program and all(c.isalnum() or c in '-_' for c in program)
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          deployment_accepted=False, **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    def wait(path, program):
        while True:
            progress = json.loads(path.read_text()) if path.exists() else None
            if progress and progress['status'] == 'completed':
                return progress
            if progress:
                assert progress['status'] not in ['failed','cancelled','aborted','cancelled-before-GPU-work'], progress
            result = subprocess.run(['supervisorctl','status',program], capture_output=True,
                                    text=True, timeout=30)
            # Supervisor returns3 during STARTING. Completion is read from the
            # producer's atomic marker before checking terminal service states.
            assert any(value in result.stdout.split() for value in ['RUNNING','STARTING']), result.stdout.strip()
            time.sleep(30)

    try:
        state('waiting-export-and-preparation')
        exported = wait(args.export_progress, args.export_program)
        prepared = wait(args.preparation_output / 'progress.json', args.preparation_program)
        assert Path(exported['artifact']).resolve() == args.artifact.resolve()
        assert prepared['correctness_inputs_pinned'] and prepared['stock_cpp_off_build_completed']
        data = json.loads((args.preparation_output / 'GMRID-input-audit.json').read_text())
        assert data['revision'] == args.data_revision and data['evaluated_items'] == 1146
        state('waiting-GPU-lock-for-artifact-audit')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
            audit = json.loads((args.artifact / 'artifact-audit.json').read_text())
            required = 16 + args.prefix_entries + 1
            assert audit['fresh_export'] and audit['checkout_and_weight_source_unchanged']
            assert audit['max_resident_sequences'] >= required
            assert audit['constant_metadata']['get_offgraph_kv_max_seqs'] == audit['max_resident_sequences']
            assert audit['supported_prefix_entries_with_16_active'] >= args.prefix_entries
            assert audit['selected_output_bound'] == [64,202048]
            assert audit['context'] == 12288 and audit['active_sessions'] == audit['decode_ceiling'] == 16
            assert audit['static_methods'] == [1,2,4,8,16,32,64]
            assert audit['GGUF_sha256'] == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            assert audit['attention_source_sha256'] == '5ce65105639785e625739c90c0b208a77d12f8544208c85aa66e143d34a4e8ea'
            assert audit['AOTI_wrapper_optimization'] == 'O2'
            assert set(audit['artifacts']) == {'model.pte','aoti_cuda_blob.ptd'}
            for name, record in audit['artifacts'].items():
                assert digest(args.artifact / name) == record['sha256']
            source = Path(__file__).resolve().parent
            wrapper = args.output / 'serve-prefix-wrapper.sh'
            subprocess.run([sys.executable,'-B',str(source / 'make_wrapper.py'),
                            '--base',str(args.base_wrapper),'--output',str(wrapper)], check=True)
            body = wrapper.read_text()
            marker = 'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-16}'
            assert body.count(marker) == 1
            assert body.count('artifact="${SB_ET_CACHEOFF_ARTIFACT:?Verified updated artifact required}"') == 1
            assert body.count('et_checkpoint_args=()') == 1
            assert body.count('"${SB_CHECKPOINT:-0}"') == 1
            body = body.replace(marker, 'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-' + str(args.prefix_entries) + '}')
            # The worker rejects positive checkpoint intervals with no cache.
            # Keep the chosen ON policy, while OFF sends an effective zero.
            body = body.replace('et_checkpoint_args=()',
                                'et_checkpoint_args=()\n'
                                'et_effective_checkpoint=${SB_CHECKPOINT:-0}\n'
                                '[ "${SB_PREFIX_CACHE:?}" != off ] || et_effective_checkpoint=0')
            body = body.replace('"${SB_CHECKPOINT:-0}"', '"$et_effective_checkpoint"')
            wrapper.write_text(body)
            subprocess.run(['bash','-n',str(wrapper)], check=True)
            os.environ.update(SB_MODEL='muse-glimmer', SB_ET_CACHEOFF_ARTIFACT=str(args.artifact.resolve()),
                              SB_ET_PREFIX_ENTRIES=str(args.prefix_entries),
                              SB_CHECKPOINT=str(args.checkpoint_interval))
            provenance = dict(artifact_audit=audit, required_resident_sequences=required,
                              prefix_entries=args.prefix_entries, wrapper_sha256=digest(wrapper),
                              checkpoint_interval=args.checkpoint_interval, cache_modes=args.cache_modes,
                              effective_checkpoint_by_cache_mode={'off':0,'on':args.checkpoint_interval},
                              total_cases=len(args.cache_modes) * len(args.concurrencies), final_best_ET_selection_pending=True,
                              source_commit_record=json.loads((source / 'source-commit.json').read_text()))
            (args.output / 'evaluation-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        total_cases = len(args.cache_modes) * len(args.concurrencies)
        state('running-matrix', total_cases=total_cases)
        command = [args.python,'-B',str(source / 'run_matrix.py'),'--wrapper',str(wrapper),
                   '--model','muse-glimmer-30b','--client',str(args.client),'--prompts',str(args.prompts),
                   '--correctness-root',str(args.preparation_output / 'correctness'),'--python',args.python,
                   '--lock',str(args.lock),'--output',str(args.output / 'matrix'),'--port',str(args.port),
                   '--variants','executorch','--cache-modes',*args.cache_modes,'--concurrencies',
                   *[str(c) for c in args.concurrencies]]
        with (args.output / 'matrix-controller.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((args.output / 'matrix/progress.json').read_text())['status'] == 'completed'
        state('completed', total_cases=total_cases,
              performance_correctness_memory_collected=True, final_best_ET_acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
