"""Evaluate a compatible Gemma prefix-capacity and write-budget candidate.

Validate the same weight, precision, selected-output and static-method contract.
Collect server performance, F1 and sampled memory, without automatic promotion.
Compare these results with the independently collected one-entry controls.
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
    parser.add_argument('--max-step-tokens', type=int, choices=[256,512], required=True)
    parser.add_argument('--prefix-entries', type=int, choices=[1,16], required=True)
    parser.add_argument('--base-wrapper', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-modes', nargs='+', choices=['off','on'], default=['on'])
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=[1,2,4,8,16], default=[1,16])
    parser.add_argument('--port', type=int, default=19085)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep prior artifacts and measurements'
    assert args.base_wrapper.is_file()
    assert args.export_program and all(c.isalnum() or c in '-_' for c in args.export_program)
    assert len(args.concurrencies) == len(set(args.concurrencies))
    args.output.mkdir(parents=True)
    completed = []

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          completed=completed, deployment_accepted=False, **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    try:
        state('waiting-artifact-export')
        while True:
            exported = json.loads(args.export_progress.read_text()) if args.export_progress.exists() else None
            if exported and exported['status'] == 'completed':
                break
            if exported:
                assert exported['status'] not in ['failed','cancelled','aborted','cancelled-before-GPU-work'], exported
            status = subprocess.run(['supervisorctl','status',args.export_program], capture_output=True,
                                    text=True, timeout=30).stdout.split()
            assert any(value in status for value in ['RUNNING','STARTING']), status
            time.sleep(30)
        assert Path(exported['artifact']).resolve() == args.artifact.resolve()
        state('waiting-GPU-lock-for-artifact-audit')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
            audit = json.loads((args.artifact / 'artifact-audit.json').read_text())
            required = 16 + args.prefix_entries + 1
            assert audit['fresh_export'] and audit['same_GGUF_and_packer']
            assert audit['runtime_and_source_checkout_unchanged']
            assert audit['max_resident_sequences'] >= required
            assert audit['supported_prefix_entries_with_16_active'] >= args.prefix_entries
            assert audit['constant_metadata']['get_offgraph_kv_max_seqs'] == audit['max_resident_sequences']
            assert audit['constant_metadata']['get_max_prefill_chunk'] == args.max_step_tokens
            assert audit['selected_output_bound'] == [64,262144]
            assert audit['static_methods'] == [1,2,4,8,16,32,64]
            assert audit['context'] == 12288 and audit['active_sessions'] == audit['decode_ceiling'] == 16
            assert audit['safe_ring_rows'] == 1024 + args.max_step_tokens - 1
            assert audit['AOTI_wrapper_optimization'] == 'O2'
            assert audit['GGUF_sha256'] == '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
            assert audit['packer_sha256'] == 'c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
            assert audit['precision_kernel_sha256'] == '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
            assert set(audit['artifacts']) == {'model.pte','aoti_cuda_blob.ptd'}
            for name, record in audit['artifacts'].items():
                assert digest(args.artifact / name) == record['sha256']
            body = args.base_wrapper.read_text()
            marker = 'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-16}'
            assert body.count(marker) == 1 and body.count('--prefix-cache-entries "$et_prefix_entries"') == 1
            pattern = re.compile(r'(?m)^([ \t]*)artifact="([^"\n]+)"[ \t]*$')
            matches = list(pattern.finditer(body))
            assert len(matches) == 1 and 'step2048' in matches[0].group(2)
            body = body.replace(marker,'et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-' + str(args.prefix_entries) + '}')
            body = pattern.sub(lambda _: matches[0].group(1) + 'artifact=' + shlex.quote(str(args.artifact.resolve())), body)
            wrapper = args.output / 'serve-policy.sh'
            wrapper.write_text(body)
            subprocess.run(['bash','-n',str(wrapper)], check=True)
            os.environ['SB_ET_PREFIX_ENTRIES'] = str(args.prefix_entries)
            provenance = dict(artifact_audit=audit, required_residents=required, prefix_entries=args.prefix_entries,
                              write_budget=args.max_step_tokens, wrapper_sha256=digest(wrapper),
                              final_ET_best_policy_selection_pending=True,
                              source_commit_record=json.loads((Path(__file__).parent/'source-commit.json').read_text()))
            (args.output/'evaluation-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        variant=f'executorch-prefix{args.prefix_entries}-step{args.max_step_tokens}'
        for cache in args.cache_modes:
            for concurrency in args.concurrencies:
                name=f'agent-{variant}-prefix-{cache}-c{concurrency}'
                state('running',current_case=name,total_cases=len(args.cache_modes)*len(args.concurrencies))
                command=[args.python,'-B',str(Path(__file__).with_name('run_case.py')),
                         '--wrapper',str(wrapper),'--backend','executorch','--variant',variant,
                         '--model','gemma4-12b-it','--prefix-cache',cache,'--client',str(args.client),
                         '--prompts',str(args.prompts),'--correctness-root',str(args.correctness_root),
                         '--python',args.python,'--concurrency',str(concurrency),'--port',str(args.port),
                         '--output',str(args.output/name),'--lock',str(args.lock)]
                with (args.output/(name+'.log')).open('x') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert json.loads((args.output/name/'progress.json').read_text())['status']=='completed'
                completed.append(name)
        state('completed',performance_correctness_memory_collected=True,final_ET_best_acceptance_pending=True)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
