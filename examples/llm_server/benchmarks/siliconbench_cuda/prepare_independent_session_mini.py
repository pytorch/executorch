"""Build and test an isolated per-session KV candidate, then export two mini arms.

No full-model export or main-result replacement is available in this tool.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--gguf', type=Path)
    parser.add_argument('--jobs', type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec_path = package / 'independent_session_capacity_source.json'
    spec = json.loads(spec_path.read_text())
    patch = package / 'independent_session_capacity.patch'
    assert digest(patch) == spec['patch_SHA256']
    assert args.python.is_file() and args.baseline_source.is_dir()
    assert not args.output.exists(), 'Preserve previous attempts'
    args.output.mkdir(parents=True)
    source = args.output / 'executorch'
    progress = args.output / 'progress.json'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', CUDAARCHS='120',
               TORCH_CUDA_ARCH_LIST='12.0', CUDA_HOME='/usr/local/cuda',
               PYTHONDONTWRITEBYTECODE='1', MAX_JOBS=str(args.jobs),
               OMP_NUM_THREADS=str(args.jobs))
    env['PATH'] = str(args.python.parent) + os.pathsep + env['PATH']
    env['PYTHONPATH'] = str(source.parent)

    def state(status, **fields):
        value = dict(status=status, controller_pid=os.getpid(), source=str(source),
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     source_spec_SHA256=digest(spec_path),
                     candidate_accepted=False, full_model_export_started=False,
                     baseline_modified=False, **fields)
        pending = progress.with_suffix('.pending.json')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(progress)
        with (args.output / 'history.jsonl').open('a') as log:
            log.write(json.dumps(value) + '\n')

    def run(argv, *, cwd=None, run_env=None, log_name='build.log', timeout=14400):
        with (args.output / log_name).open('a') as log:
            log.write(json.dumps({'argv': [str(v) for v in argv], 'cwd': str(cwd)}) + '\n')
            log.flush()
            subprocess.run([str(v) for v in argv], cwd=cwd, env=run_env or env,
                           stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    def verify(root, version):
        for name, record in spec['files'].items():
            expected = record[version + '_SHA256']
            p = root / name
            if expected is None:
                assert not p.exists(), name
            else:
                assert digest(p) == expected, name

    try:
        state('verifying-baseline-source')
        verify(args.baseline_source, 'before')
        controls = {}
        for name in ['backends/cuda/triton/kernels/w4a8_gemm.py',
                     'backends/cuda/triton/kernels/w4a16_gemm.py',
                     'extension/cuda/cuda_allocator.cpp']:
            p = args.baseline_source / name
            if p.is_file():
                controls[name] = digest(p)
        state('copying-isolated-source')

        def ignore(directory, names):
            return [name for name in names if name in
                    ['.git', 'cmake-out', '__pycache__', '.pytest_cache'] or
                    name.endswith(('.pte', '.ptd', '.gguf', '.log', '.nsys-rep'))]

        shutil.copytree(args.baseline_source, source, symlinks=True, ignore=ignore)
        verify(source, 'before')
        run(['git', 'apply', '--check', patch], cwd=source)
        run(['git', 'apply', patch], cwd=source)
        verify(source, 'after')
        for name, expected in controls.items():
            assert digest(source / name) == expected, name
        cmake = shutil.which('cmake', path=env['PATH'])
        assert cmake is not None
        prefix = subprocess.check_output(
            [args.python, '-B', '-c', 'import torch; print(torch.utils.cmake_prefix_path)'],
            env=env, text=True).strip()
        build = source / 'cmake-out'
        state('building-native-core', unchanged_controls=controls)
        run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DEXECUTORCH_BUILD_PRESET_FILE=' + str(source / 'tools/cmake/preset/llm.cmake'),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DEXECUTORCH_BUILD_PYBIND=OFF',
             '-DEXECUTORCH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON',
             '-DCMAKE_INSTALL_PREFIX=' + str(build), '-DCMAKE_PREFIX_PATH=' + prefix,
             '-DPYTHON_EXECUTABLE=' + str(args.python)])
        run([cmake, '--build', build, '--parallel', args.jobs,
             '--target', 'test_cuda_kv_cache', 'install'])
        model_build = args.output / 'model-build'
        state('building-mini-runtime-probe')
        run([cmake, '-S', source / 'examples/models/muse-glimmer', '-B', model_build,
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + str(build),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DMUSE_GLIMMER_BUILD_BATCHING=ON',
             '-DMUSE_GLIMMER_BUILD_RUNTIME_KV_PROBE=ON'])
        run([cmake, '--build', model_build, '--parallel', args.jobs,
             '--target', 'muse_glimmer_runtime_kv_probe', 'muse_glimmer_batching_worker'])
        binary = build / 'backends/cuda/test_cuda_kv_cache'
        binaries = {str(p): digest(p) for p in [binary,
                    model_build / 'muse_glimmer_runtime_kv_probe',
                    model_build / 'muse_glimmer_batching_worker']}
        verify(source, 'after')
        verify(args.baseline_source, 'before')
        state('waiting-native-GPU-tests', binaries=binaries)
        xml = args.output / 'native-tests.xml'
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            gpu_env = dict(env, CUDA_VISIBLE_DEVICES='0')
            state('running-native-GPU-tests', binaries=binaries)
            run([binary, '--gtest_output=xml:' + str(xml)], run_env=gpu_env,
                log_name='native-tests.log', timeout=1800)
        tests = list(ET.parse(xml).getroot().iter('testcase'))
        assert len(tests) >= 29
        assert not any(t.find('failure') is not None or t.find('error') is not None or
                       t.find('skipped') is not None for t in tests)
        independent = [t.attrib['name'] for t in tests
                       if t.attrib['classname'] == 'CudaIndependentSessionCacheTest']
        assert len(independent) == 7
        receipt = dict(native_GPU_tests_pass=True, test_count=len(tests),
                       independent_session_tests=independent,
                       source_spec_SHA256=digest(spec_path), source=str(source),
                       binaries=binaries, unchanged_controls=controls,
                       native_tests_xml_SHA256=digest(xml),
                       model_numerics_pass=False, model_performance_pass=False,
                       full_model_export_authorized_by_this_receipt=False)
        receipt_path = args.output / 'native-gate.json'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        state('native-GPU-tests-passed', native_gate=str(receipt_path),
              test_count=len(tests), independent_session_tests=independent)
        if args.gguf:
            for role in ['mini-global', 'mini-independent']:
                state('exporting-' + role, native_gate=str(receipt_path))
                run([args.python, '-B', '-u', package / 'export_independent_session_mini.py',
                     '--source', source, '--native-gate', receipt_path, '--gguf', args.gguf,
                     '--role', role, '--output', args.output / (role + '-artifact'),
                     '--job-dir', args.output / (role + '-export'), '--lock', args.lock],
                    run_env=dict(env, CUDA_VISIBLE_DEVICES='0'),
                    log_name=role + '-controller.log', timeout=21600)
        state('mini-artifacts-ready' if args.gguf else 'native-GPU-tests-passed',
              native_gate=str(receipt_path), next_gate='mini model numerical, graph and performance qualification')
    except Exception as error:
        state('failed', error_type=type(error).__name__, detail=str(error)[-2000:],
              traceback=traceback.format_exc()[-2500:])
        raise


if __name__ == '__main__':
    main()
