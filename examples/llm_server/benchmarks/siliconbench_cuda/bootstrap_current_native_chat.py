"""Collect the remaining stock 128K chat comparisons after an owned GPU job.

Use the existing benchmark client and four-round case runner. Framework
sources, existing environments, existing model files, and prior results are
read only. Full agent F1 evidence is reused; destination smoke is separate.
"""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

CPP_COMMIT = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
TASKS = {
    'MG-cpp': ('muse-glimmer', 'llamacpp'),
    'Gemma-cpp': ('gemma4', 'llamacpp'),
    'Gemma-vllm': ('gemma4', 'vllm'),
    'Gemma-sglang': ('gemma4', 'sglang'),
}
MODELS = {
    'muse-glimmer': {
        'alias': 'muse-glimmer-30b', 'input_tokens': 100893,
        'gguf_repo': 'meta-models/Muse-Glimmer-30B-GGUF',
        'gguf_revision': 'b44f1072cf75e918a1b82a3a17b45c4389b076e7',
        'gguf_file': 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf',
        'gguf_SHA256': '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e',
        'prefill': 4096,
    },
    'gemma4': {
        'alias': 'gemma4-12b-it', 'input_tokens': 99128,
        'gguf_repo': 'unsloth/gemma-4-12b-it-GGUF',
        'gguf_revision': 'fc034cfff751157913579611efad8462ac1be606',
        'gguf_file': 'gemma-4-12b-it-Q4_0.gguf',
        'gguf_SHA256': '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a',
        'HF_repo': 'Danny-Dasilva/gemma-4-12B-it-W4A16-GPTQ-g32-DSpark',
        'HF_revision': '03c395f3cdf0b93add98571b5dda842f4de1a759',
        'config_SHA256': '47e9440ed6e0dfd4c3c06abf1b3a64dacac6119e940bb7411565712395ca5137',
        'prefill': 2048,
    },
}
MINI_TERMINAL = {
    'mini-performance-signal-kept-no-full-deployment',
    'mini-qualified-pending-full-correctness-and-performance',
    'mini-gate-not-passed-no-full-deployment',
    'failed-preserve-original-experiments',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_gguf(model, directory, reuse=None):
    spec = MODELS[model]
    target = reuse or directory/spec['gguf_file']
    if reuse:
        assert target.is_file() and sha(target) == spec['gguf_SHA256']
    else:
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name+'.partial')
        URL = f"https://huggingface.co/{spec['gguf_repo']}/resolve/{spec['gguf_revision']}/{spec['gguf_file']}"
        with urllib.request.urlopen(URL, timeout=120) as source, pending.open('xb') as out:
            for block in iter(lambda: source.read(16 * 1024 * 1024), b''):
                out.write(block)
        assert sha(pending) == spec['gguf_SHA256']
        pending.replace(target)
    return {'path': str(target), 'SHA256': spec['gguf_SHA256'], 'existing_file_read_only': bool(reuse)}


def download_gemma(directory):
    from huggingface_hub import HfApi, snapshot_download
    spec = MODELS['gemma4']
    assert not directory.exists()
    directory.mkdir(parents=True)
    info = HfApi().model_info(spec['HF_repo'], revision=spec['HF_revision'], files_metadata=True)
    assert info.sha == spec['HF_revision']
    snapshot_download(repo_id=spec['HF_repo'], revision=spec['HF_revision'], local_dir=directory,
        max_workers=4, allow_patterns=['*.safetensors', '*.safetensors.index.json', '*.json', '*.jinja', 'README.md'])
    cfg = json.loads((directory/'config.json').read_text())
    assert sha(directory/'config.json') == spec['config_SHA256']
    quant = cfg['quantization_config']
    assert quant['quant_method'] == 'compressed-tensors' and quant['format'] == 'pack-quantized'
    for group in quant['config_groups'].values():
        weight = group['weights']
        assert weight['num_bits'] == 4 and weight['group_size'] == 32 and weight['symmetric']
        assert not group.get('input_activations') and not group.get('output_activations')
    files = {}
    for sibling in info.siblings:
        file = directory/sibling.rfilename
        assert file.resolve().is_relative_to(directory.resolve())
        if not file.is_file():
            continue
        actual = sha(file)
        lfs = sibling.lfs
        expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if file.suffix == '.safetensors':
            assert actual == expected, sibling.rfilename
        files[sibling.rfilename] = {'bytes': file.stat().st_size, 'SHA256': actual, 'public_LFS_SHA256': expected}
    assert any(name.endswith('.safetensors') for name in files)
    proof = dict(repo=spec['HF_repo'], revision=spec['HF_revision'], quantization=quant,
                 files=files, no_requantization=True, existing_models_modified=False)
    (directory/'download-audit.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


def write_wrapper(path, model, backend, tokenizer, weight, executable, cache):
    spec = MODELS[model]
    port = 'ENV_PORT'
    prefix = 'ENV_PREFIX'
    kwargs = {'enable_thinking': False}
    if model == 'muse-glimmer':
        kwargs = {'reasoning_strength': 'high', 'current_date': '2026-10-21'}
    if backend == 'llamacpp':
        argv = [str(executable), '--model', str(weight), '--alias', spec['alias'], '--host', '127.0.0.1',
                '--port', port, '--n-gpu-layers', '99', '--parallel', '16', '--ctx-size', '131072',
                '--kv-unified', '--kv-unified-per-slot', '131072', '--jinja', '--chat-template-file',
                str(tokenizer/'chat_template.jinja'), '--chat-template-kwargs', json.dumps(kwargs),
                '--skip-chat-parsing', '--no-prefill-assistant', '--batch-size', str(spec['prefill']),
                '--ubatch-size', str(spec['prefill']), '--cache-type-k', 'bf16', '--cache-type-v', 'bf16',
                prefix, '--flash-attn', 'on', '--threads', '8', '--threads-batch', '8']
        if model == 'gemma4':
            argv += ['--logit-bias', '258882-inf', '--logit-bias', '258883-inf']
        on = '--cache-prompt --cache-ram 8192 --cache-reuse 0'
        off = '--no-cache-prompt --cache-ram 0 --cache-reuse 0'
    elif backend == 'vllm':
        vllm_bin = executable.parent/'vllm'
        assert vllm_bin.is_file()
        argv = [str(vllm_bin), 'serve', str(weight),
                '--served-model-name', spec['alias'], '--tokenizer', str(tokenizer), '--language-model-only',
                '--quantization', 'compressed-tensors', '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
                '--host', '127.0.0.1', '--port', port, '--max-model-len', '131072', '--max-num-seqs', '16',
                '--max-num-batched-tokens', str(spec['prefill']), prefix,
                '--generation-config', 'auto', '--chat-template-content-format', 'string',
                '--compilation-config', json.dumps({'cudagraph_mode':'FULL_AND_PIECEWISE','cudagraph_capture_sizes':[1,2,4,8,16]})]
        on, off = '--enable-prefix-caching', '--no-enable-prefix-caching'
    else:
        argv = [str(executable), '-B', '-m', 'sglang.launch_server', prefix, '--model-path', str(weight),
                '--tokenizer-path', str(tokenizer), '--served-model-name', spec['alias'],
                '--json-model-override-args', json.dumps({'vision_config':None,'audio_config':None,'language_model_only':True}),
                '--host', '127.0.0.1', '--port', port, '--context-length', '131072', '--max-running-requests', '16',
                '--quantization', 'compressed-tensors', '--dtype', 'bfloat16',
                '--kv-cache-dtype', 'bf16', '--chunked-prefill-size', '2048', '--max-prefill-tokens', '2048',
                '--cuda-graph-max-bs-decode', '16', '--cuda-graph-backend-prefill', 'breakable',
                '--cuda-graph-max-bs-prefill', '512', '--default-chat-template-kwargs',
                json.dumps({'enable_thinking':False,'current_date':'2026-10-21','reasoning_strength':'high'})]
        on, off = '', '--disable-radix-cache'
    rendered = ' '.join('"${SB_PORT:?}"' if x == port else '"${cache_args[@]}"' if x == prefix else shlex.quote(x) for x in argv)
    short = Path('/tmp')/('sb-chat-'+str(os.getpid())+'-'+backend)
    short.mkdir(exist_ok=True)
    cache.mkdir()
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', f'[ "${{1:?}}" = {shlex.quote(backend)} ]',
             'export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1',
             'unset PYTHONPATH SB_ET_PREFILL_TRACE SB_PREFILL_A16 SB_PREFILL_TRACE_DIR GGML_CUDA_CUBLAS_COMPUTE_TYPE',
             'export CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas',
             'export VLLM_PLUGINS="" VLLM_USE_V2_MODEL_RUNNER=1',
             'export TMPDIR='+shlex.quote(str(short)),
             'export TRITON_CACHE_DIR='+shlex.quote(str(cache/'triton')),
             'export TORCHINDUCTOR_CACHE_DIR='+shlex.quote(str(cache/'inductor')),
             'export FLASHINFER_WORKSPACE_BASE='+shlex.quote(str(cache/'flashinfer')),
             'export HF_HOME='+shlex.quote(str(cache/'hf')),
             'export XDG_CACHE_HOME='+shlex.quote(str(cache/'xdg')),
             'export CUDA_CACHE_PATH='+shlex.quote(str(cache/'cuda')),
             'case "${SB_PREFIX_CACHE:?}" in',
             f' on) cache_args=({on}) ;;', f' off) cache_args=({off}) ;;', ' *) exit 2 ;;', 'esac',
             'exec '+rendered, '']
    path.write_text('\n'.join(lines))
    subprocess.run(['bash', '-n', str(path)], check=True)
    return {'model': model, 'backend': backend, 'argv_template': argv, 'prefix_ON_explicit_args': on,
            'prefix_OFF_explicit_args': off, 'wrapper_SHA256': sha(path), 'prefill_budget': spec['prefill'],
            'logical_context': 131072, 'active_sessions': 16, 'decode_limit': 16, 'BF16_KV': True,
            'stock_framework_sources_modified': False, 'FORCE_CUBLAS': False if backend == 'llamacpp' else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), required=True)
    p.add_argument('--after-job', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--expected-GPU-UUID', required=True)
    p.add_argument('--power-W', type=int, required=True)
    p.add_argument('--cpp-source', type=Path)
    p.add_argument('--cpp-AGENTS-SHA256')
    p.add_argument('--reuse-MG-gguf', type=Path)
    p.add_argument('--vllm-python', type=Path)
    p.add_argument('--sglang-python', type=Path)
    a = p.parse_args()
    assert len(a.tasks) == len(set(a.tasks)) and not a.output.exists()
    a.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    started = time.monotonic()
    completed = []
    def state(status, **extra):
        path = a.output/'progress.pending.json'
        path.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
            tasks=a.tasks, completed=completed, accepted=False, **extra), indent=2)+'\n')
        path.replace(a.output/'progress.json')
    def run(label, argv, timeout):
        (a.output/(label+'-argv.json')).write_text(json.dumps(list(map(str, argv)), indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(list(map(str, argv)), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
    try:
        while True:
            prior = json.loads((a.after_job/'progress.json').read_text())
            pid = prior['controller_pid']
            cmd = Path('/proc')/str(pid)/'cmdline'
            live = cmd.is_file() and bool(cmd.read_bytes())
            if prior['status'] in MINI_TERMINAL and not live:
                break
            assert live, 'Prior handle missing without authoritative terminal evidence; do not restart it'
            assert b'bootstrap_workspace_preserving_contraction_mini.py' in cmd.read_bytes()
            assert str(a.after_job).encode() in cmd.read_bytes()
            state('waiting-confirmed-live-prior-mini-no-build-or-GPU', prior_PID=pid, prior_status=prior['status'])
            time.sleep(30)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
            state('waiting-existing-GPU-work-no-build'); time.sleep(15)
        GPU = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,power.limit','--format=csv,noheader'], text=True).strip()
        assert GPU.split(',')[0] == a.expected_GPU_UUID
        assert float(GPU.split(',')[-1].split()[0]) == a.power_W
        assert sha(a.inputs/'chat_benchmark_prompts.json') == '3bd0cb44d6719032c4019a49df611cda2b3e0d7694affda3b47fdfc304ffeddd'
        benchmark = a.inputs/'.venv/bin/python'
        assert benchmark.is_file()
        assert sha(package/'benchmark_mg.py') == '3a748babe2fc62d9f10c08892de65b57954d9661bd0689988ac3d6d5cbabf982'
        assert sha(package/'benchmark_gemma.py') == 'd2f9eba743b583b9b5968d2f8704cb347638a05fe12717797d21a8c938a00b05'
        state('preparing-pinned-public-assets-after-prior-mini', GPU=GPU)
        models = list(dict.fromkeys(TASKS[t][0] for t in a.tasks))
        assets = {}
        for model in models:
            tokenizer_job = a.output/(model+'-canonical-tokenizer')
            helper = 'prepare_mg_canonical_tokenizer.py' if model == 'muse-glimmer' else 'prepare_gemma_canonical_tokenizer.py'
            run(model+'-tokenizer', [sys.executable,'-B',package/helper,'--output',tokenizer_job], 900)
            assets[model] = {'tokenizer': tokenizer_job/'tokenizer'}
            if any(TASKS[t] == (model,'llamacpp') for t in a.tasks):
                assets[model]['GGUF'] = download_gguf(model, a.output/'models', a.reuse_MG_gguf if model == 'muse-glimmer' else None)
        if any(TASKS[t][1] in ['vllm','sglang'] for t in a.tasks):
            native_model = a.output/'models/gemma-native-group32'
            proof = download_gemma(native_model)
            assets['gemma4']['HF'] = {'path':str(native_model),'download_audit_SHA256':sha(native_model/'download-audit.json')}
        cpp_binary = None
        cpp_proof = None
        if any(TASKS[t][1] == 'llamacpp' for t in a.tasks):
            assert a.cpp_source and subprocess.check_output(['git','-C',str(a.cpp_source),'rev-parse','HEAD'],text=True).strip() == CPP_COMMIT
            assert not subprocess.check_output(['git','-C',str(a.cpp_source),'status','--porcelain','--untracked-files=no'],text=True).strip()
            agents = a.cpp_source/'AGENTS.md'
            assert not agents.exists() or sha(agents) == a.cpp_AGENTS_SHA256
            build = a.output/'CPP-server-build'
            run('configure-stock-default-CPP', ['cmake','-S',a.cpp_source,'-B',build,'-DGGML_CUDA=ON','-DGGML_CUDA_FORCE_CUBLAS=OFF',
                '-DGGML_CUDA_FORCE_MMQ=OFF','-DGGML_CUDA_GRAPHS=ON','-DCMAKE_CUDA_ARCHITECTURES=120',
                '-DCMAKE_BUILD_TYPE=Release','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'], 900)
            run('build-stock-default-CPP', ['cmake','--build',build,'--target','llama-server','--parallel','4'], 7200)
            cpp_binary = build/'bin/llama-server'
            cache = (build/'CMakeCache.txt').read_text()
            assert 'GGML_CUDA_FORCE_CUBLAS:BOOL=OFF' in cache and 'GGML_CUDA:BOOL=ON' in cache
            cpp_proof = dict(commit=CPP_COMMIT, binary=str(cpp_binary), binary_SHA256=sha(cpp_binary),
                             CMakeCache_SHA256=sha(build/'CMakeCache.txt'), source_read_only=True, FORCE_CUBLAS=False)
        task_records = {}
        for task in a.tasks:
            model, backend = TASKS[task]
            executable = cpp_binary if backend == 'llamacpp' else a.vllm_python if backend == 'vllm' else a.sglang_python
            assert executable and executable.is_file()
            if backend != 'llamacpp':
                required = {'torch':'2.13.0' if backend=='vllm' else '2.14.1','transformers':'5.19.0',backend:'0.32.0' if backend=='vllm' else '0.5.22'}
                run(task+'-installed-versions', [executable,'-B','-c',
                    "import importlib.metadata as m,json;v={n:m.version(n) for n in " + repr(list(required)) + "};print(json.dumps(v));assert v=="+repr(required)], 120)
            wrapper = a.output/(task+'-serve-stock.sh')
            weight = Path(assets[model]['GGUF']['path']) if backend == 'llamacpp' else Path(assets[model]['HF']['path'])
            task_records[task] = write_wrapper(wrapper,model,backend,assets[model]['tokenizer'],weight,executable,a.output/(task+'-runtime-cache'))
            client = package/('benchmark_mg.py' if model=='muse-glimmer' else 'benchmark_gemma.py')
            task_records[task].update(client_SHA256=sha(client), CPP=cpp_proof if backend=='llamacpp' else None)
            for mode in ['correctness','performance']:
                for prefix in ['off','on']:
                    for c in ([16,1] if mode=='correctness' else [16,1,8,4,2]):
                        split = 'agent' if mode=='correctness' else 'chat'
                        name = f'{task}-{mode}-{split}-{prefix}-c{c}'
                        case = a.output/name
                        argv = [benchmark,'-B',package/'run_case_24h.py','--wrapper',wrapper,'--backend',backend,
                            '--model',MODELS[model]['alias'],'--variant','current128K-stock-'+task,'--prefix-cache',prefix,
                            '--client',client,'--prompts',a.inputs/(split+'_benchmark_prompts.json'),
                            '--correctness-root',a.inputs/'correctness','--python',benchmark,'--concurrency',c,
                            '--port',a.port,'--output',case,'--lock',a.lock,'--mode',mode,'--split',split]
                        if mode=='correctness':
                            argv += ['--quality-scope','smoke','--correctness-max-tokens','1024',
                                '--correctness-chat-template-kwargs',json.dumps({'enable_thinking':False,'reasoning_strength':'low','current_date':'2026-10-21'})]
                        state('collecting-'+mode, GPU=GPU, current_case=str(case), task=task)
                        run(name,argv,43200)
                        assert json.loads((case/'progress.json').read_text())['status']=='completed'
                        if mode=='correctness':
                            from run_preserved_MG_pair_after_main import review_smoke
                            review_smoke(case,a.inputs/'correctness',.80)
                        else:
                            aggregate = json.loads((case/'performance-aggregate.json').read_text())
                            assert aggregate['successful']['values']==[100,100,100] and aggregate['failed']['values']==[0,0,0]
                            assert aggregate['total_input_tokens']['values']==[MODELS[model]['input_tokens']]*3
                        completed.append(name)
                        (a.output/'provenance.json').write_text(json.dumps(dict(GPU=GPU,tasks=task_records,
                            assets={m:{k:str(v) if isinstance(v,Path) else v for k,v in x.items()} for m,x in assets.items()},
                            full_agent_F1_reused_not_rerun=True,destination_smoke_separate=True,prior_experiments_modified=False,
                            source_SHA256={name:sha(package/name) for name in ['bootstrap_current_native_chat.py','run_case_24h.py','benchmark_mg.py','benchmark_gemma.py']}),indent=2)+'\n')
        assert len(completed)==14*len(a.tasks)
        result = dict(performance_cells=10*len(a.tasks),destination_smoke_cells=4*len(a.tasks),completed=completed,
                      independent_raw_review_pending=True,accepted=False,GPU=GPU)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('current128K-stock-chat-collected-pending-independent-review',GPU=GPU)
    except BaseException as error:
        state('failed-preserve-existing-experiments',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
