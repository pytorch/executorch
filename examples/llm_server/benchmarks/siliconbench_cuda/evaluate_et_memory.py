"""Compare an audited smaller-write artifact against the original ET artifact.

The parent owns no CUDA context. Each server case independently holds the shared
GPU lock through teardown. Preserve every artifact, result and failed attempt;
collect performance, memory and task F1 without automatically promoting a model.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
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
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-export-progress', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--prefix-entries', type=int, choices=[1,16], default=1)
    parser.add_argument('--base-wrapper', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19083)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep previous results and failed attempts'
    assert args.export_progress.exists() and args.baseline_export_progress.exists() and args.base_wrapper.is_file()
    args.output.mkdir(parents=True)
    completed = []

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          completed=completed, **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    try:
        state('waiting-artifact-export', export_progress=str(args.export_progress),
              baseline_export_progress=str(args.baseline_export_progress))
        while True:
            exports = [json.loads(path.read_text()) for path in
                       [args.baseline_export_progress, args.export_progress]]
            if all(export['status'] == 'completed' for export in exports):
                break
            assert all(export['status'] not in ['failed', 'cancelled', 'aborted'] for export in exports), exports
            time.sleep(30)
        assert [Path(export['artifact']).resolve() for export in exports] == [args.baseline.resolve(), args.candidate.resolve()]
        state('waiting-GPU-lock-for-artifact-audit')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
            audit = json.loads((args.candidate / 'artifact-audit.json').read_text())
            baseline_audit = json.loads((args.baseline / 'artifact-audit.json').read_text())
            required_resident = 16 + args.prefix_entries + 1
            for artifact, proof, budget in [(args.baseline, baseline_audit, 2048),
                                            (args.candidate, audit, 512)]:
                assert proof['max_resident_sequences'] >= required_resident
                assert proof['constant_metadata']['get_offgraph_kv_max_seqs'] == proof['max_resident_sequences']
                assert proof['constant_metadata']['get_max_prefill_chunk'] == budget
                assert proof['supported_prefix_entries_with_16_active'] >= args.prefix_entries
                assert proof['fresh_export'] and proof['same_GGUF_and_packer']
                assert proof['runtime_and_source_checkout_unchanged']
                assert proof['selected_output_bound'] == [64, 262144]
                assert proof['context'] == 12288 and proof['active_sessions'] == proof['decode_ceiling'] == 16
                assert proof['static_methods'] == [1, 2, 4, 8, 16, 32, 64]
                assert proof['AOTI_wrapper_optimization'] == 'O2'
                assert set(proof['artifacts']) == {'model.pte', 'aoti_cuda_blob.ptd'}
                for name, record in proof['artifacts'].items():
                    assert digest(artifact / name) == record['sha256']
            assert baseline_audit['GGUF_sha256'] == audit['GGUF_sha256']
            assert baseline_audit['packer_sha256'] == audit['packer_sha256']
            assert baseline_audit['precision_kernel_sha256'] == audit['precision_kernel_sha256']
            assert baseline_audit['max_resident_sequences'] == audit['max_resident_sequences']
            assert audit['fresh_export'] and audit['same_GGUF_and_packer']
            assert audit['runtime_and_source_checkout_unchanged']
            assert audit['GGUF_sha256'] == '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
            assert audit['precision_kernel_sha256'] == '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
            assert audit['selected_output_bound'] == [64, 262144]
            assert audit['context'] == 12288 and audit['active_sessions'] == audit['decode_ceiling'] == 16
            assert audit['static_methods'] == [1, 2, 4, 8, 16, 32, 64]
            assert audit['constant_metadata']['get_max_prefill_chunk'] == 512
            assert audit['safe_ring_rows'] == 1535
            assert audit['AOTI_wrapper_optimization'] == 'O2'
            body = args.base_wrapper.read_text()
            pattern = re.compile(r'(?m)^([ \t]*)artifact="([^"\n]+)"[ \t]*$')
            matches = list(pattern.finditer(body))
            assert len(matches) == 1, 'Expected one reviewed ET artifact assignment'
            assert 'step2048' in matches[0].group(2), 'Expected the existing 2048-write baseline'
            assert body.count('--prefix-cache-entries "$et_prefix_entries"') == 1
            entry_marker = 'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-16}'
            assert body.count(entry_marker) == 1
            body = body.replace(entry_marker, 'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-' + str(args.prefix_entries) + '}')
            os.environ['SB_ET_PREFIX_ENTRIES'] = str(args.prefix_entries)
            replacement = matches[0].group(1) + 'artifact=' + shlex.quote(str(args.candidate.resolve()))
            candidate_body = pattern.sub(lambda _: replacement, body)
            base_wrapper = args.output / 'serve-baseline.sh'
            candidate_wrapper = args.output / 'serve-candidate.sh'
            base_wrapper.write_text(pattern.sub(lambda _: matches[0].group(1) + 'artifact=' + shlex.quote(str(args.baseline.resolve())), body))
            candidate_wrapper.write_text(candidate_body)
            for wrapper in [base_wrapper, candidate_wrapper]:
                subprocess.run(['bash', '-n', str(wrapper)], check=True)
            provenance = dict(candidate_audit=audit, baseline_audit=baseline_audit,
                              prefix_entries=args.prefix_entries, required_resident_sequences=required_resident,
                              candidate_wrapper_sha256=digest(candidate_wrapper),
                              base_wrapper_sha256=digest(base_wrapper), total_cases=8,
                              only_wrapper_change='ET artifact path', deployment_accepted=False)
            (args.output / 'comparison-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        for cache in ['off', 'on']:
            for concurrency in [1, 16]:
                for variant, wrapper in [('executorch-step2048', base_wrapper),
                                         ('executorch-step512', candidate_wrapper)]:
                    name = f'agent-{variant}-prefix-{cache}-c{concurrency}'
                    state('running', current_case=name, total_cases=8)
                    command = [args.python, '-B', str(Path(__file__).with_name('run_case.py')),
                               '--wrapper', str(wrapper), '--backend', 'executorch',
                               '--variant', variant, '--model', 'gemma4-12b-it',
                               '--prefix-cache', cache, '--client', str(args.client),
                               '--prompts', str(args.prompts), '--correctness-root', str(args.correctness_root),
                               '--python', args.python, '--concurrency', str(concurrency),
                               '--port', str(args.port), '--output', str(args.output / name),
                               '--lock', str(args.lock)]
                    with (args.output / (name + '.log')).open('x') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    assert json.loads((args.output / name / 'progress.json').read_text())['status'] == 'completed'
                    completed.append(name)
        state('completed', total_cases=8, performance_correctness_memory_collected=True,
              final_memory_quality_cache_and_best_ET_acceptance_pending=True,
              no_automatic_runtime_replacement=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
