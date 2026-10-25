"""Prepare an isolated roofline shard and run one paired operator screen.

Uses the pinned public checkout, official wheels and exact public GGUF.
Run under supervisor; existing environments and model directories are untouched.
The final full-forward comparison is a separate job on one common GPU.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import traceback
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--role', choices=['ffn_gate_up', 'attention_out'], required=True)
    p.add_argument('--attempt', default='bootstrap-v2')
    a = p.parse_args()
    assert '/' not in a.attempt and a.attempt not in ['', '.', '..']
    job = a.root / a.attempt
    job.mkdir(exist_ok=False)
    state_lock = threading.Lock()
    status = dict(controller_pid=os.getpid(), role=a.role, phases={}, public_only=True,
                  existing_environments_untouched=True)

    def state(phase, value, **extra):
        with state_lock:
            status['phases'][phase] = value
            status.update(extra, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            tmp = job / 'progress.pending.json'
            tmp.write_text(json.dumps(status, indent=2) + '\n'); tmp.replace(job / 'progress.json')

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_HOME='/usr/local/cuda',
               TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', CUDA_VISIBLE_DEVICES='',
               MAX_JOBS='8', CMAKE_BUILD_PARALLEL_LEVEL='8', OMP_NUM_THREADS='2',
               UV_CACHE_DIR=str(a.root / 'uv-cache'))
    venv = a.root / 'venv'
    python = venv / 'bin/python'
    uv = shutil.which('uv'); assert uv

    def run(command, timeout=3600):
        with (job / 'install.log').open('a') as log:
            log.write(json.dumps([str(v) for v in command]) + '\n'); log.flush()
            subprocess.run([str(v) for v in command], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def setup():
        state('environment', 'installing-isolated-wheels')
        if not python.exists():
            run([uv, 'venv', '--python', '/usr/bin/python3', venv])
        run([uv, 'pip', 'install', '--python', python, '--index-url',
             'https://download.pytorch.org/whl/cu132', 'torch==2.14.0+cu132'])
        run([uv, 'pip', 'install', '--python', python, '--no-deps',
             '--index-url', 'https://download.pytorch.org/whl/nightly/cu132',
             'torchao==0.19.0.dev20260907+cu132'])
        run([uv, 'pip', 'install', '--python', python, '--index-url',
             'https://pypi.org/simple', 'transformers==5.20.2', 'gguf==0.19.0',
             'numpy==2.5.4', 'safetensors==0.8.0', 'huggingface-hub==2.2.0',
             'flatbuffers==25.12.19', 'cmake', 'ninja', 'setuptools', 'wheel',
             'packaging', 'ruamel.yaml', 'pyyaml', 'sympy', 'tqdm', 'parameterized',
             'expecttest', 'antlr4-python3-runtime==4.9.3'])
        state('environment', 'building-public-python-package')
        run(['git', '-C', a.source, 'submodule', 'update', '--init', '--recursive', '--depth', '1'], 5400)
        run([uv, 'pip', 'install', '--python', python, '--no-deps', '--no-build-isolation',
             '--editable', a.source], 5400)
        run([python, '-B', '-c', 'import torch,triton;from executorch.backends.cuda.triton.kernels import int4_quantized_gemm;from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda;print(torch.__version__,triton.__version__)'])
        state('environment', 'ready')

    name = 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf'
    model = a.root / 'models' / name
    def download():
        model.parent.mkdir(exist_ok=True)
        if model.exists():
            digest = hashlib.sha256()
            with model.open('rb') as source:
                while block := source.read(16 * 1024 * 1024): digest.update(block)
            assert digest.hexdigest() == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            state('model', 'reused-verified', model_bytes=model.stat().st_size,
                  GGUF_SHA256=digest.hexdigest())
            return
        url = ('https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/'
               'b44f1072cf75e918a1b82a3a17b45c4389b076e7/' + name)
        state('model', 'downloading-public-pinned-GGUF')
        digest = hashlib.sha256(); total = 0
        partial = model.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('xb') as output:
            while block := response.read(16 * 1024 * 1024):
                output.write(block); digest.update(block); total += len(block)
                if total % (512 * 1024 * 1024) == 0:
                    state('model', 'downloading-public-pinned-GGUF', model_bytes=total)
        assert digest.hexdigest() == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
        partial.replace(model)
        state('model', 'ready', model_bytes=total, GGUF_SHA256=digest.hexdigest())

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(setup), pool.submit(download)]
            for future in futures: future.result()
        state('screen', 'running-operator-autotune')
        env['CUDA_VISIBLE_DEVICES'] = '0'
        lock = a.root / 'gpu-benchmark.lock'; lock.touch(exist_ok=True)
        result = a.root / ('operator-' + a.role + '-' + a.attempt)
        command = [python, '-B', Path(__file__).with_name('benchmark_roofline_large_w4a16.py'),
                   '--et-source', a.source, '--gguf', model, '--output', result, '--lock', lock,
                   '--roles', a.role, '--M', '64', '72', '128', '256', '512', '520', '1024',
                   '--max-seconds', '14400']
        run(command, 15000)
        state('screen', 'completed-pending-common-GPU-full-forward-validation')
    except BaseException as error:
        state('controller', 'failed', error=str(error), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
