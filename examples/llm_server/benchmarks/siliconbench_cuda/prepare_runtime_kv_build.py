"""Prepare and compile an isolated runtime-column candidate without GPU work.

Existing source, environments, models, binaries and results remain untouched.
CUDA devices are hidden for the entire build. This script never runs tests,
exports a model, starts a server, or accepts the candidate.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec = json.loads((package / 'runtime_kv_columns_source.json').read_text())
    patch = package / 'runtime_kv_columns.patch'
    assert digest(patch) == spec['patch_sha256']
    assert args.baseline_source.is_dir() and args.python.is_file()
    assert not args.output.exists(), 'Preserve every prior build and attempt'
    args.output.mkdir(parents=True)
    source = args.output / 'executorch'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TORCH_CUDA_ARCH_LIST='12.0',
               CUDAARCHS='120', CUDA_HOME='/usr/local/cuda',
               PYTHONDONTWRITEBYTECODE='1', MAX_JOBS=str(args.jobs),
               OMP_NUM_THREADS=str(args.jobs))
    env['PATH'] = str(args.python.parent) + os.pathsep + env['PATH']
    progress = args.output / 'progress.json'

    def state(status, **fields):
        value = dict(status=status, controller_pid=os.getpid(),
                     updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     CUDA_visible_devices='', GPU_tests_started=False,
                     GPU_model_export_started=False, candidate_accepted=False,
                     baseline_modified=False, **fields)
        temp = progress.with_suffix('.pending.json')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(progress)

    def run(argv, *, cwd=None, timeout=7200):
        with (args.output / 'build.log').open('a') as log:
            log.write(json.dumps(dict(argv=[str(a) for a in argv], cwd=str(cwd))) + '\n')
            log.flush()
            subprocess.run([str(a) for a in argv], cwd=cwd, env=env,
                           stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    def git(root, *argv):
        return subprocess.check_output(['git', '-C', str(root), *argv],
                                       text=True, env=env).strip()

    try:
        state('isolating-source')
        assert git(args.baseline_source, 'rev-parse', 'HEAD') == spec['base_commit']
        baseline_status = git(args.baseline_source, 'status', '--porcelain')
        # The existing checkout is a partial clone. Packing its entire local
        # history can request absent promisor objects; fetch the public base
        # into an independent clone instead of altering that repository.
        run(['git', 'clone', '--filter=blob:none', '--depth=1', '--no-checkout',
             'https://github.com/pytorch/executorch.git', source])
        run(['git', '-C', source, 'fetch', '--depth=1', 'origin', spec['base_commit']])
        run(['git', '-C', source, 'checkout', '--detach', spec['base_commit']])
        # Preserve current task fixes outside the candidate diff, e.g. the
        # validated allocator policy. Copy tracked source only, never artifacts.
        inherited = {}
        for name in git(args.baseline_source, 'diff', '--name-only').splitlines():
            if name in spec['changed_files']:
                expected = spec['selected_baseline_sha256'].get(
                    name, spec['changed_files'][name]['before_sha256'])
                assert digest(args.baseline_source / name) == expected, name
                continue
            assert name.startswith(('backends/cuda/', 'extension/cuda/',
                                    'extension/llm/', 'examples/models/muse-glimmer/',
                                    'examples/llm_server/')), name
            path = Path(name)
            assert path.suffix in ('.py', '.cpp', '.h', '.cu', '.cmake') or path.name == 'CMakeLists.txt', name
            assert (args.baseline_source / name).is_file()
            shutil.copy2(args.baseline_source / name, source / name)
            inherited[name] = digest(source / name)
        for name, record in spec['changed_files'].items():
            if record['before_sha256'] is not None:
                assert digest(source / name) == record['before_sha256'], name
        run(['git', '-C', source, 'apply', '--check', patch])
        run(['git', '-C', source, 'apply', patch])
        for name, record in spec['changed_files'].items():
            assert digest(source / name) == record['after_sha256'], name
        controls = {}
        for name in spec['unchanged_quant_files']:
            assert digest(source / name) == digest(args.baseline_source / name), name
            controls[name] = digest(source / name)
        run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive'])
        assert git(args.baseline_source, 'status', '--porcelain') == baseline_status
        # The unchanged Python binding is only used for later export. Its
        # native ABI is not changed by this patch; no environment is installed.
        binding_code = 'import executorch.extension.pybindings._portable_lib as p; print(p.__file__)'
        baseline_env = env.copy()
        baseline_env['PYTHONPATH'] = str(args.baseline_source.parent)
        binding = Path(subprocess.check_output([str(args.python), '-B', '-c', binding_code],
                                               env=baseline_env, text=True).strip())
        binding_dest = source / 'extension/pybindings' / binding.name
        assert not binding_dest.exists()
        binding_dest.symlink_to(binding)
        torch_prefix = subprocess.check_output(
            [str(args.python), '-B', '-c', 'import torch; print(torch.utils.cmake_prefix_path)'],
            env=env, text=True).strip()
        cmake = Path(shutil.which('cmake', path=env['PATH']))
        build = source / 'cmake-out'
        env['PYTHONPATH'] = str(source.parent)
        state('building-core', source=str(source), inherited_source=inherited,
              unchanged_quant_source=controls)
        run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DEXECUTORCH_BUILD_PRESET_FILE=' + str(source / 'tools/cmake/preset/llm.cmake'),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DEXECUTORCH_BUILD_PYBIND=OFF',
             '-DEXECUTORCH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON',
             '-DCMAKE_INSTALL_PREFIX=' + str(build), '-DCMAKE_PREFIX_PATH=' + torch_prefix,
             '-DPYTHON_EXECUTABLE=' + str(args.python)])
        run([cmake, '--build', build, '--parallel', args.jobs,
             '--target', 'test_cuda_kv_cache', 'install'])
        model_build = args.output / 'muse-glimmer-build'
        state('building-probe', source=str(source), inherited_source=inherited,
              unchanged_quant_source=controls)
        run([cmake, '-S', source / 'examples/models/muse-glimmer', '-B', model_build,
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + str(build),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DMUSE_GLIMMER_BUILD_BATCHING=ON',
             '-DMUSE_GLIMMER_BUILD_RUNTIME_KV_PROBE=ON'])
        run([cmake, '--build', model_build, '--parallel', args.jobs,
             '--target', 'muse_glimmer_runtime_kv_probe', 'muse_glimmer_batching_worker'])
        binaries = [build / 'backends/cuda/test_cuda_kv_cache',
                    model_build / 'muse_glimmer_runtime_kv_probe',
                    model_build / 'muse_glimmer_batching_worker']
        assert all(path.is_file() for path in binaries)
        assert git(args.baseline_source, 'status', '--porcelain') == baseline_status
        state('CPU-build-completed', source=str(source), inherited_source=inherited,
              unchanged_quant_source=controls, patch_sha256=spec['patch_sha256'],
              binaries={str(path): digest(path) for path in binaries},
              next_gate='All 100 main performance cells, 40 full and 60 smoke correctness cells independently accepted before any GPU test/export')
    except Exception as error:
        state('failed', error_type=type(error).__name__, detail=str(error)[-1600:])
        raise


if __name__ == '__main__':
    main()
