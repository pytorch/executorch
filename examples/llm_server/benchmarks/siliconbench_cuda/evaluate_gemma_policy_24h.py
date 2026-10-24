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
    parser.add_argument('--checkpoint-interval', type=int, choices=[0,512,1024], default=0)
    parser.add_argument('--checkpoint-preparation', type=Path)
    parser.add_argument('--checkpoint-program')
    parser.add_argument('--initial-kv-capacity', type=int, choices=[512,1024,2048,12288], default=12288)
    parser.add_argument('--base-wrapper', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-modes', nargs='+', choices=['off','on'], default=['on'])
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=[1,2,4,8,16], default=[1,16])
    parser.add_argument('--phases', nargs='+', choices=['performance','correctness'], default=['performance','correctness'])
    parser.add_argument('--port', type=int, default=19085)
    parser.add_argument('--max-context', type=int, choices=[12288,131072], default=12288)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep prior artifacts and measurements'
    assert args.base_wrapper.is_file()
    assert args.export_program and all(c.isalnum() or c in '-_' for c in args.export_program)
    assert len(args.concurrencies) == len(set(args.concurrencies))
    if args.checkpoint_interval:
        assert args.checkpoint_preparation and args.checkpoint_program
    if args.checkpoint_preparation:
        assert args.checkpoint_program and all(c.isalnum() or c in '-_' for c in args.checkpoint_program)
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
        checkpoint_build = None
        if args.checkpoint_preparation:
            state('waiting-checkpoint-worker-preparation')
            progress_path = args.checkpoint_preparation/'progress.json'
            while True:
                prepared = json.loads(progress_path.read_text()) if progress_path.exists() else None
                if prepared and prepared['status'] == 'completed':
                    break
                if prepared:
                    assert prepared['status'] not in ['failed','cancelled','aborted'], prepared
                status = subprocess.run(['supervisorctl','status',args.checkpoint_program], capture_output=True,
                                        text=True, timeout=30).stdout.split()
                assert any(value in status for value in ['RUNNING','STARTING']), status
                time.sleep(30)
            checkpoint_build = json.loads((args.checkpoint_preparation/'build-provenance.json').read_text())
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
            assert audit['context'] == args.max_context and audit['active_sessions'] == audit['decode_ceiling'] == 16
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
            assert body.count('--initial-kv-capacity 12288') == 1
            body = body.replace('--initial-kv-capacity 12288',
                                '--initial-kv-capacity '+str(args.initial_kv_capacity))
            if checkpoint_build:
                assert checkpoint_build['original_source_and_libraries_unchanged']
                assert checkpoint_build['static_graphs_enabled_by_default']
                assert checkpoint_build['existing_shared_checkpoint_runtime']
                assert checkpoint_build['original_raw_streaming_and_tool_text_preserved']
                binary = Path(checkpoint_build['worker_binary'])
                launcher = Path(checkpoint_build['launcher'])
                assert digest(binary) == checkpoint_build['binary_sha256']
                assert digest(launcher) == checkpoint_build['launcher_sha256']
                launch = re.compile(r'(?m)^(\s*exec "\$r/\.venvs/executorch/bin/python" -B )([^\n]+?)( \\\s*)$')
                assert len(list(launch.finditer(body))) == 1
                body = launch.sub(lambda m: m[1]+shlex.quote(str(launcher.resolve()))+m[3], body)
                old_worker = '--worker-bin "$r/runtime/gemma4_batching_worker"'
                assert body.count(old_worker) == 1
                body = body.replace(old_worker,'--worker-bin '+shlex.quote(str(binary.resolve())))
                cache_flag = '--prefix-cache-entries "$et_prefix_entries"'
                assert body.count(cache_flag) == 1
                body = body.replace(cache_flag,cache_flag+' --prefix-checkpoint-interval "$et_effective_checkpoint"')
                guard = 'et_effective_checkpoint='+str(args.checkpoint_interval)+'\n'
                guard += '[ "${SB_PREFIX_CACHE:?}" != off ] || et_effective_checkpoint=0\n'
                assert body.count('case "$backend" in') == 1
                body = body.replace('case "$backend" in',guard+'case "$backend" in')
            wrapper = args.output / 'serve-policy.sh'
            if args.max_context != 12288:
                from configure_context import configure
                body = configure(body, args.max_context, args.initial_kv_capacity, args.max_context)
            wrapper.write_text(body)
            subprocess.run(['bash','-n',str(wrapper)], check=True)
            os.environ['SB_ET_PREFIX_ENTRIES'] = str(args.prefix_entries)
            provenance = dict(artifact_audit=audit, required_residents=required, prefix_entries=args.prefix_entries,
                              write_budget=args.max_step_tokens, wrapper_sha256=digest(wrapper),
                              checkpoint_interval=args.checkpoint_interval, checkpoint_worker=checkpoint_build,
                              effective_checkpoint_by_cache_mode={'off':0,'on':args.checkpoint_interval},
                              initial_kv_capacity=args.initial_kv_capacity, maximum_context=args.max_context, phases=args.phases,
                              final_ET_best_policy_selection_pending=True,
                              source_commit_record=json.loads((Path(__file__).parent/'source-commit.json').read_text()))
            (args.output/'evaluation-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        variant=f'executorch-prefix{args.prefix_entries}-step{args.max_step_tokens}'
        if checkpoint_build:
            variant += f'-checkpoint{args.checkpoint_interval}'
        if args.initial_kv_capacity != 12288:
            variant += f'-initialkv{args.initial_kv_capacity}'
        for phase in args.phases:
            for cache in args.cache_modes:
                for concurrency in args.concurrencies:
                    name=f'{phase}-agent-{variant}-prefix-{cache}-c{concurrency}'
                    state('running',current_case=name,total_cases=len(args.cache_modes)*len(args.concurrencies))
                    command=[args.python,'-B',str(Path(__file__).with_name('run_case_24h.py')),
                             '--mode',phase,'--quality-scope','full' if concurrency in [1,16] else 'smoke',
                             '--wrapper',str(wrapper),'--backend','executorch','--variant',variant,
                             '--model','gemma4-12b-it','--prefix-cache',cache,'--client',str(args.client),
                             '--prompts',str(args.prompts),'--correctness-root',str(args.correctness_root),
                             '--python',args.python,'--concurrency',str(concurrency),'--port',str(args.port),
                             '--output',str(args.output/name),'--lock',str(args.lock)]
                    with (args.output/(name+'.log')).open('x') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    assert json.loads((args.output/name/'progress.json').read_text())['status']=='completed'
                    completed.append(name)
        state('completed',performance_correctness_memory_collected=set(args.phases)=={'performance','correctness'},final_ET_best_acceptance_pending=True)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
