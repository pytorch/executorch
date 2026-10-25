"""Build the Gemma runtime-column overlay in a new isolated CPU-only source.

This prepares an idle maintenance hook and preserves the existing Gemma
attention launch decisions. It does not run a model or qualify the unfinished
low-peak column migration policy.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-source', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    package = Path(__file__).resolve().parent
    spec = json.loads((package / 'gemma_runtime_columns_overlay.json').read_text())
    base = json.loads((package / 'runtime_kv_columns_source.json').read_text())
    patch = package / 'gemma_runtime_columns_overlay.patch'
    assert sha(patch) == spec['overlay_SHA256']
    assert not a.output.exists(), 'Preserve previous sources and build attempts'
    a.output.mkdir(parents=True)
    source = a.output / 'executorch'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDAARCHS='120', TORCH_CUDA_ARCH_LIST='12.0',
               CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS=str(a.jobs), MAX_JOBS=str(a.jobs),
               PYTHONDONTWRITEBYTECODE='1')
    env['PATH'] = str(a.python.parent) + os.pathsep + env['PATH']
    progress = a.output / 'progress.json'

    def state(status, **fields):
        value = dict(status=status, controller_pid=os.getpid(), source=str(source),
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     GPU_work=False, model_exported=False, model_promoted=False,
                     low_peak_migration_qualification_pending=True, **fields)
        pending = progress.with_suffix('.pending.json')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(progress)

    def run(argv, timeout=7200):
        with (a.output / 'build.log').open('a') as log:
            log.write(json.dumps([str(v) for v in argv]) + '\n')
            log.flush()
            subprocess.run([str(v) for v in argv], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    try:
        state('copying-isolated-runtime-source')
        original = {name: sha(a.runtime_source / name) for name in base['changed_files']}
        assert all(original[name] == record['after_sha256'] for name, record in base['changed_files'].items())
        shutil.copytree(a.runtime_source, source,
                        ignore=shutil.ignore_patterns('cmake-out', '.git', '__pycache__', '*.pte', '*.ptd', '*.gguf'))
        # These are exact task source baselines, not weights or evaluation data.
        baseline_files = {'extension/llm/batching/runner.cpp': 'gemma_runtime_runner_baseline.cpp',
                          'examples/models/gemma4_31b/export_batching.py': 'gemma_runtime_export_baseline.py'}
        for target, baseline in baseline_files.items():
            dest = source / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / baseline, dest)
        for name, record in spec['files'].items():
            assert sha(source / name) == record['before_SHA256'], name
        run(['git', '-C', source, 'apply', '--check', patch.resolve()])
        run(['git', '-C', source, 'apply', patch.resolve()])
        for name, record in spec['files'].items():
            assert sha(source / name) == record['after_SHA256'], name
        assert original == {name: sha(a.runtime_source / name) for name in original}
        torch_prefix = subprocess.check_output([str(a.python), '-B', '-c',
                                               'import torch;print(torch.utils.cmake_prefix_path)'],
                                              env=env, text=True).strip()
        env['PYTHONPATH'] = str(source.parent)
        build = source / 'cmake-out'
        state('building-isolated-core', overlay_SHA256=sha(patch))
        run(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DEXECUTORCH_BUILD_PRESET_FILE=' + str(source / 'tools/cmake/preset/llm.cmake'),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DEXECUTORCH_BUILD_PYBIND=OFF',
             '-DEXECUTORCH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON',
             '-DCMAKE_INSTALL_PREFIX=' + str(build), '-DCMAKE_PREFIX_PATH=' + torch_prefix,
             '-DPYTHON_EXECUTABLE=' + str(a.python)])
        run(['cmake', '--build', build, '--parallel', a.jobs, '--target', 'test_cuda_kv_cache', 'install'])
        assert original == {name: sha(a.runtime_source / name) for name in original}
        state('CPU-core-build-completed', overlay_SHA256=sha(patch),
              source_files={name: sha(source / name) for name in spec['files']},
              baseline_source_unchanged=True, runtime_test_binary=str(build / 'backends/cuda/test_cuda_kv_cache'))
    except Exception as error:
        state('failed', error_type=type(error).__name__, error=str(error)[-1800:])
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
