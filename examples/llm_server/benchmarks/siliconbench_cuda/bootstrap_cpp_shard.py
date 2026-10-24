"""Prepare a dedicated stock CUDA 12.8 llama.cpp comparison shard.

All downloads are pinned public sources. Model and tokenizer hashes must match
those supplied by the caller. Only the caller's new task directory is written.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['muse-glimmer', 'gemma4'], required=True)
    p.add_argument('--gguf-repo', required=True)
    p.add_argument('--gguf-revision', required=True)
    p.add_argument('--gguf-file', required=True)
    p.add_argument('--gguf-sha256', required=True)
    p.add_argument('--template-repo', required=True)
    p.add_argument('--template-revision', required=True)
    p.add_argument('--template-sha256', required=True)
    p.add_argument('--data-revision', required=True)
    p.add_argument('--resume-reviewed-setup', action='store_true')
    p.add_argument('--repo-instructions-sha256')
    p.add_argument('--port', type=int, default=19082)
    args = p.parse_args()
    r = args.root
    if args.resume_reviewed_setup:
        previous = json.loads((r/'progress.json').read_text())
        assert previous['status']=='failed' and previous['detail']=='Read repository instructions before building'
        archive = r/'setup-instructions-review-stop.json'
        assert not archive.exists()
        archive.write_text(json.dumps(previous, indent=2)+'\n')
    else:
        assert not r.exists(), 'Use a fresh task directory; preserve existing work'
        r.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    logs = r/'logs'
    logs.mkdir(exist_ok=args.resume_reviewed_setup)

    def state(status, **fields):
        pending = r/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                         utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields), indent=2)+'\n')
        pending.replace(r/'progress.json')

    def run(command, name, timeout=7200):
        with (logs/(name+'.log')).open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)

    def download(url, path, expected=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name+'.partial')
        assert not path.exists() and not pending.exists()
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, pending.open('xb') as output:
            while block := response.read(8*1024*1024):
                digest.update(block)
                output.write(block)
        actual = digest.hexdigest()
        assert expected is None or actual == expected, 'Public file differs from the comparison source'
        pending.replace(path)
        return dict(url=url, sha256=actual, bytes=path.stat().st_size)

    try:
        state('preparing-public-stock-shard', GPU_work_started=False)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        python = r/'.venv/bin/python'
        repo = r/'llama.cpp'
        commit = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
        if not args.resume_reviewed_setup:
            run(['uv', 'venv', '--python', '3.12', str(r/'.venv')], 'venv', 120)
            run(['uv', 'pip', 'install', '--python', str(python), 'aiohttp>=3.9', 'scikit-learn>=1.3'], 'dependencies', 600)
            run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/ggml-org/llama.cpp.git', str(repo)], 'clone', 600)
            run(['git', '-C', str(repo), 'fetch', '--depth=1', 'origin', commit], 'fetch', 600)
            run(['git', '-C', str(repo), 'checkout', '--detach', commit], 'checkout', 120)
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == commit
        instructions = repo/'AGENTS.md'
        if instructions.exists():
            assert args.repo_instructions_sha256 == hashlib.sha256(instructions.read_bytes()).hexdigest(), 'Read repository instructions before building'
        gguf = r/'models'/args.gguf_file
        template = r/'models/chat_template.jinja'
        prompts = r/'agent_benchmark_prompts.json'
        downloads = [
            (f'https://huggingface.co/{args.gguf_repo}/resolve/{args.gguf_revision}/{args.gguf_file}', gguf, args.gguf_sha256),
            (f'https://huggingface.co/{args.template_repo}/resolve/{args.template_revision}/chat_template.jinja', template, args.template_sha256),
            ('https://huggingface.co/datasets/windchimeran/SiliconBench/resolve/3e9fa20c6fc6de333c5cb40e5da46d001194443c/agent_benchmark_prompts.json', prompts, None),
        ]
        build_options = ['-DGGML_CUDA=ON', '-DGGML_CUDA_FORCE_MMQ=OFF', '-DGGML_CUDA_GRAPHS=ON',
                         '-DCMAKE_CUDA_ARCHITECTURES=120', '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_CURL=OFF']
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(download, *item) for item in downloads]
            for name, flag in [('build-stock-cublas', 'ON'), ('build-stock-native-evaluation', 'OFF')]:
                run(['cmake', '-S', str(repo), '-B', str(repo/name), *build_options,
                     '-DGGML_CUDA_FORCE_CUBLAS='+flag], 'configure-'+flag, 600)
                run(['cmake', '--build', str(repo/name), '--target', 'llama-server', '-j', '8'], 'build-'+flag, 7200)
            proofs = [f.result() for f in futures]
        (r/'public-inputs.json').write_text(json.dumps(proofs, indent=2)+'\n')
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        prep = r/'correctness-preparation'
        run([str(python), '-B', str(source/'prepare.py'), '--root', str(r), '--cpp-repo', str(repo),
             '--output', str(prep), '--python', str(python), '--reuse-build', '--data-revision', args.data_revision], 'correctness-inputs', 1200)
        is_mg = args.model == 'muse-glimmer'
        alias = 'muse-glimmer-30b' if is_mg else 'gemma4-12b-it'
        kwargs = '{"reasoning_strength":"high","current_date":"2026-10-21"}' if is_mg else '{"enable_thinking":false}'
        base = [str(repo/'${SB_CPP_BUILD}/bin/llama-server'), '--model', str(gguf), '--alias', alias,
                '--host', '127.0.0.1', '--port', '${SB_PORT:-19082}', '--n-gpu-layers', '99',
                '--parallel', '16', '--ctx-size', '131072', '--kv-unified', '--kv-unified-per-slot', '131072',
                '--jinja', '--chat-template-file', str(template), '--chat-template-kwargs', kwargs,
                '--skip-chat-parsing', '--no-prefill-assistant', '--batch-size', '8192' if is_mg else '2048',
                '--ubatch-size', '8192' if is_mg else '2048', '--cache-type-k', 'bf16', '--cache-type-v', 'bf16',
                '${cache_args[@]}', '--flash-attn', 'on', '--threads', '8', '--threads-batch', '8']
        if not is_mg:
            base += ['--logit-bias', '258882-inf', '--logit-bias', '258883-inf']
        rendered = ' '.join('"'+x+'"' if '${' in x else shlex.quote(x) for x in base)
        wrapper = r/'serve-public-cpp.sh'
        wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
[ "${1:?}" = llamacpp ]
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
unset SB_PREFILL_A16 SB_ET_PREFILL_TRACE GGML_CUDA_CUBLAS_COMPUTE_TYPE
SB_CPP_BUILD=${SB_CPP_BUILD:-build-stock-cublas}
case "${SB_PREFIX_CACHE:?}" in
 on) cache_args=(--cache-prompt --cache-ram 8192 --cache-reuse 0) ;;
 off) cache_args=(--no-cache-prompt --cache-ram 0 --cache-reuse 0) ;;
 *) exit 2 ;;
esac
exec '''+rendered+'\n')
        run(['bash', '-n', str(wrapper)], 'wrapper-syntax', 30)
        audit = dict(model=args.model, stock_commit=commit, runtime_code_modified=False,
                     CUDA_toolkit=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
                     GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit', '--format=csv,noheader'], text=True).strip(),
                     maximum_context=131072, CPP_shared_KV_token_pool=131072, active_sessions=16,
                     build_cache_hashes={name: hashlib.sha256((repo/name/'CMakeCache.txt').read_bytes()).hexdigest()
                        for name in ['build-stock-cublas', 'build-stock-native-evaluation']}, inputs=proofs)
        (r/'deployment-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        state('running-performance-first-matrix', performance_cells=20, full_quality_cells=8, smoke_quality_cells=12)
        run([str(python), '-B', str(source/'run_matrix_24h.py'), '--wrapper', str(wrapper), '--model', alias,
             '--client', str(source/('benchmark_mg.py' if is_mg else 'benchmark_gemma.py')),
             '--prompts', str(prompts), '--correctness-root', str(prep/'correctness'), '--python', str(python),
             '--output', str(r/'matrix'), '--lock', str(r/'GPU-benchmark.lock'), '--port', str(args.port),
             '--variants', 'llamacpp-off', 'llamacpp-on', '--cache-modes', 'off', 'on',
             '--concurrencies', '1', '2', '4', '8', '16'], 'matrix', 172800)
        state('completed', acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
