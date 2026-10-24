"""Reuse a completed stock CPP shard for isolated MG correctness work."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = a.existing_root
    assert json.loads((r/'matrix/progress.json').read_text())['status'] == 'completed'
    assert not a.output.exists(), 'Keep every existing model and result'
    a.output.mkdir(parents=True)

    def state(status, **fields):
        pending = a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, **fields), indent=2)+'\n')
        pending.replace(a.output/'progress.json')

    try:
        state('preparing-public-MG-inputs', GPU_work_started=False)
        repo = r/'llama.cpp'
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == '620d2225cc436ab0b001e3200e590e3b286ef9ab'
        assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
        builds = {}
        for build, flag in [('build-stock-cublas','ON'),('build-stock-native-evaluation','OFF')]:
            cache = repo/build/'CMakeCache.txt'
            text = cache.read_text()
            for setting in ['GGML_CUDA:BOOL=ON','GGML_CUDA_GRAPHS:BOOL=ON','GGML_CUDA_FORCE_MMQ:BOOL=OFF','GGML_CUDA_FORCE_CUBLAS:BOOL='+flag]:
                assert setting in text, setting
            assert (repo/build/'bin/llama-server').is_file()
            builds[build] = {'cache_sha256':digest(cache),'binary_sha256':digest(repo/build/'bin/llama-server')}
        inputs = [
            ('https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/b44f1072cf75e918a1b82a3a17b45c4389b076e7/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf', 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf', '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'),
            ('https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/cec94fcfd6512f2cfe708138c32533b346ea4f69/chat_template.jinja', 'chat_template.jinja', 'cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678'),
        ]
        proofs = []
        for url, name, expected in inputs:
            target = a.output/name
            pending = target.with_name(name+'.partial')
            with urllib.request.urlopen(url,timeout=120) as response, pending.open('xb') as output:
                while block := response.read(8*1024*1024):
                    output.write(block)
            assert digest(pending) == expected, 'Model/template differs from main MG comparison'
            pending.replace(target)
            proofs.append({'url':url,'sha256':expected,'bytes':target.stat().st_size})
        model = a.output/inputs[0][1]
        template = a.output/inputs[1][1]
        command = [str(repo/'${SB_CPP_BUILD}/bin/llama-server'),'--model',str(model),
                   '--alias','muse-glimmer-30b','--host','127.0.0.1','--port','${SB_PORT:-19091}',
                   '--n-gpu-layers','99','--parallel','16','--ctx-size','131072',
                   '--kv-unified','--kv-unified-per-slot','131072','--jinja',
                   '--chat-template-file',str(template),'--chat-template-kwargs',
                   '{"reasoning_strength":"high","current_date":"2026-10-21"}',
                   '--skip-chat-parsing','--no-prefill-assistant','--batch-size','8192',
                   '--ubatch-size','8192','--cache-type-k','bf16','--cache-type-v','bf16',
                   '${cache_args[@]}','--flash-attn','on','--threads','8','--threads-batch','8']
        rendered = ' '.join('"'+x+'"' if '${' in x else shlex.quote(x) for x in command)
        wrapper = a.output/'serve-MG-quality.sh'
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
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        receipt = dict(stock_source_unchanged=True,existing_models_and_results_unchanged=True,
                       stock_commit='620d2225cc436ab0b001e3200e590e3b286ef9ab',builds=builds,
                       inputs=proofs,wrapper_sha256=digest(wrapper),
                       correctness_root=str(r/'correctness-preparation/correctness'),
                       context=131072,active_sessions=16,BF16_KV=True,output_budget=256,
                       existing_main_performance_stays_on_original_host=True,
                       performance_work_started=False)
        (a.output/'reuse-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
        state('completed',**receipt)
    except Exception as e:
        state('failed',error_type=type(e).__name__,detail=str(e)[-1200:])
        raise


if __name__ == '__main__':
    main()
