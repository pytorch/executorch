"""Prepare stock CUDA serving and replace invalid MG chat checkpoint results.

Download the pinned public INT4/group32 checkpoint, install official packages
in a separate venv, qualify destination HTTP responses, then collect chat with
the unchanged SiliconBench client. Existing environments and results are read
only. Full agent correctness/performance is reused separately, not claimed here.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


REPO = 'cyankiwi/Muse-Glimmer-30B-AWQ-INT4'
REVISION = '55c08e88f6bf8e9785e28713beca21c9c5b55695'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(16 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def download_model(model):
    from huggingface_hub import HfApi, snapshot_download
    model.mkdir(parents=True, exist_ok=False)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    card = info.card_data.to_dict()
    assert card['base_model'] == 'meta-models/Muse-Glimmer-30B'
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=model, max_workers=4,
        allow_patterns=['*.safetensors', '*.safetensors.index.json', 'config.json', 'README.md',
                        'generation_config.json', 'tokenizer*.json', 'special_tokens_map.json', 'chat_template.jinja'])
    cfg = json.loads((model / 'config.json').read_text())
    assert cfg['architectures'] == ['MuseGlimmerForConditionalGeneration']
    quant = cfg['quantization_config']
    assert quant['quant_method'] == 'compressed-tensors' and quant['format'] == 'pack-quantized'
    assert quant['config_groups']
    for group in quant['config_groups'].values():
        w = group['weights']
        assert w['num_bits'] == 4 and w['type'] == 'int' and w['group_size'] == 32 and not w['symmetric']
        assert not group.get('input_activations') and not group.get('output_activations')
    files = {}
    for sibling in info.siblings:
        path = model / sibling.rfilename
        if not path.is_file():
            continue
        actual = sha(path)
        lfs = sibling.lfs
        expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if path.suffix == '.safetensors':
            assert expected == actual, sibling.rfilename
        files[sibling.rfilename] = dict(bytes=path.stat().st_size, SHA256=actual, public_LFS_SHA256=expected)
    assert any(name.endswith('.safetensors') for name in files)
    proof = dict(repo=REPO, revision=REVISION, base_model=card['base_model'], quantization=quant,
                 files=files, original_downloaded_weight_bytes_unchanged=True, no_requantization=True)
    (model / 'download-audit.json').write_text(json.dumps(proof, indent=2) + '\n')
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('vllm', 'sglang'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-venv', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--expected-GPU-UUID', required=True)
    p.add_argument('--target-power-W', type=int, default=575)
    a = p.parse_args()
    assert not a.output.exists() and not a.native_venv.exists() and not a.model_path.exists()
    a.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    started = time.monotonic()
    def state(status, **extra):
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), backend=a.backend,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
            repo=REPO, revision=REVISION, accepted=False, **extra), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')
    def run(label, argv, timeout, env=None):
        (a.output / (label + '-command.json')).write_text(json.dumps(dict(argv=list(map(str, argv))), indent=2) + '\n')
        with (a.output / (label + '.log')).open('x') as log:
            subprocess.run(list(map(str, argv)), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout, env=env)
    try:
        uv = shutil.which('uv')
        assert uv and Path('/usr/local/cuda/bin/ptxas').is_file()
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Preserve an existing GPU task'
        before = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,power.limit', '--format=csv,noheader'], text=True).strip()
        assert before.split(',')[0] == a.expected_GPU_UUID
        if float(before.split(',')[1].strip().split()[0]) != a.target_power_W:
            run('align-stock-GPU-power', ['nvidia-smi', '-pl', a.target_power_W], 30)
        GPU = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit', '--format=csv,noheader'], text=True).strip()
        state('installing-stock-native-and-downloading-approved-group32', GPU=GPU)
        cache = a.native_venv.parent / (a.native_venv.name + '-installer-cache')
        env = dict(os.environ, UV_CACHE_DIR=str(cache), UV_LINK_MODE='copy', PYTHONDONTWRITEBYTECODE='1',
                   CUDA_VISIBLE_DEVICES='', MAX_JOBS='4')
        def install():
            run('native-venv', [uv, 'venv', '--python', '3.12', a.native_venv], 300, env)
            native = a.native_venv / 'bin/python'
            requirements = ['vllm==0.32.0', 'torch==2.13.0', 'transformers==5.19.0'] if a.backend == 'vllm' else ['sglang==0.5.22', 'torch==2.14.1', 'transformers==5.19.0']
            run('official-native-packages', [uv, 'pip', 'install', '--python', native, *requirements], 1800, env)
            run('native-version-and-CUDA', [native, '-B', '-c',
                "import importlib.metadata as m,json,torch;print(json.dumps({'versions':{n:m.version(n) for n in ['torch','transformers','" + a.backend + "']},'CUDA':torch.version.cuda}));assert torch.version.cuda.startswith('13.')"], 120, env)
            run('official-package-freeze', [uv, 'pip', 'freeze', '--python', native], 120, env)
            return native
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            model_future = pool.submit(download_model, a.model_path)
            install_future = pool.submit(install)
            native = install_future.result()
            model_proof = model_future.result()
        state('preparing-canonical-tokenizer-and-frozen-inputs', GPU=GPU)
        run('canonical-tokenizer', [sys.executable, '-B', package / 'prepare_mg_canonical_tokenizer.py', '--output', a.output / 'canonical-tokenizer'], 900)
        inputs = a.inputs or a.output / 'inputs'
        if not a.inputs:
            run('pinned-inputs', [sys.executable, '-B', package / 'prepare_independent_main_public_inputs.py', '--output', inputs], 900)
        assert sha(inputs / 'chat_benchmark_prompts.json') == '3bd0cb44d6719032c4019a49df611cda2b3e0d7694affda3b47fdfc304ffeddd'
        assert sha(package / 'benchmark_mg.py') == '3a748babe2fc62d9f10c08892de65b57954d9661bd0689988ac3d6d5cbabf982'
        assert (inputs / '.venv/bin/python').is_file()
        wrapper = a.output / 'serve-stock-group32.sh'
        run('stock-wrapper', [sys.executable, '-B', package / 'make_mg_stock_quality_wrapper.py',
            '--model-path', a.model_path, '--tokenizer-path', a.output / 'canonical-tokenizer/tokenizer',
            '--vllm-bin', a.native_venv / 'bin/vllm' if a.backend == 'vllm' else native,
            '--sglang-python', native, '--task-root', a.output / 'runtime-cache', '--output', wrapper], 120)
        # AF_UNIX paths in stock SGLang must fit its native limit. Only change
        # the exposed TMPDIR storage path, as in the validated stock recipe.
        if a.backend == 'sglang':
            short = Path('/tmp') / ('sb-MG-chat-' + str(os.getpid()))
            short.mkdir(exist_ok=False)
            text = wrapper.read_text()
            old = 'TMPDIR="$task/tmp"'
            assert text.count(old) == 1
            import shlex
            wrapper.write_text(text.replace(old, 'TMPDIR=' + shlex.quote(str(short))))
        run('wrapper-syntax', ['bash', '-n', wrapper], 30)
        provenance = dict(GPU=GPU, power_before=before, model_proof=model_proof,
            repo=REPO, revision=REVISION, framework=a.backend, native_python=str(native),
            package_freeze_SHA256=sha(a.output / 'official-package-freeze.log'),
            wrapper_SHA256=sha(wrapper), client_SHA256=sha(package / 'benchmark_mg.py'),
            prompt_SHA256=sha(inputs / 'chat_benchmark_prompts.json'), stock_runtime_code_unmodified=True,
            W4A16_BF16_KV=True, native_prefill_budget=8192, active_decode_limit=16, logical_context=131072,
            original_complete_agent_F1_and_performance_reused_separately=True,
            destination_full_agent_F1_not_claimed=True, all_new_results_require_independent_raw_review=True)
        (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        benchmark = inputs / '.venv/bin/python'
        completed = []
        for mode in ('correctness', 'performance'):
            split = 'agent' if mode == 'correctness' else 'chat'
            for prefix in ('off', 'on'):
                for c in ((16, 1) if mode == 'correctness' else (16, 1, 8, 4, 2)):
                    case = a.output / (mode + '-' + split + '-' + prefix + '-c' + str(c))
                    argv = [benchmark, '-B', package / 'run_case_24h.py', '--wrapper', wrapper,
                        '--backend', a.backend, '--model', 'muse-glimmer-30b', '--variant', 'approved-native-group32-chat',
                        '--prefix-cache', prefix, '--client', package / 'benchmark_mg.py', '--prompts', inputs / (split + '_benchmark_prompts.json'),
                        '--correctness-root', inputs / 'correctness', '--python', benchmark, '--concurrency', c,
                        '--port', a.port, '--output', case, '--lock', a.lock, '--mode', mode, '--split', split]
                    if mode == 'correctness':
                        argv += ['--quality-scope', 'smoke', '--correctness-rounds', '1', '--correctness-max-tokens', '1024',
                                 '--correctness-chat-template-kwargs', json.dumps(dict(enable_thinking=False, current_date='2026-10-21', reasoning_strength='low'))]
                    state('collecting-' + mode, GPU=GPU, case=str(case), completed=completed)
                    run(case.name, argv, 43200)
                    assert json.loads((case / 'progress.json').read_text())['status'] == 'completed'
                    if mode == 'correctness':
                        from run_preserved_MG_pair_after_main import review_smoke
                        review_smoke(case, inputs / 'correctness', .80)
                    else:
                        aggregate = json.loads((case / 'performance-aggregate.json').read_text())
                        assert aggregate['successful']['mean'] == 100 and aggregate['failed']['mean'] == 0
                    completed.append(case.name)
        assert len(completed) == 14
        (a.output / 'result.json').write_text(json.dumps(dict(completed=completed, chat_performance_cases=10,
            destination_portability_smoke_cases=4, repo=REPO, revision=REVISION, accepted=False,
            independent_raw_review_pending=True, elapsed_s=time.monotonic()-started), indent=2) + '\n')
        state('corrected-native-group32-chat-collected-pending-independent-review', GPU=GPU, completed=completed)
    except BaseException as error:
        state('failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
