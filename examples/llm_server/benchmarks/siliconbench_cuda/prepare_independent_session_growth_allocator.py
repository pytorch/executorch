"""Build allocator-growth control in an isolated copy of a qualified ABI3 SDK.

Retain allocator cached blocks during growth. Preserve eager cached release on
contraction. Existing mini PTEs/PTDs and original SDK/source remain untouched.
No model/kernel/ABI/sliding change or full-model export is available.
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
    parser.add_argument('--mini-preparation-job',type=Path,
                        help='Optional exact mini artifacts to alias; omit for native SDK preparation only')
    parser.add_argument('--baseline-native-gate',type=Path,required=True)
    parser.add_argument('--stable-fastpath',action='store_true',
                        help='Also avoid repeated layout table construction when every owner and capacity is unchanged')
    parser.add_argument('--jobs', type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec_path = package / 'independent_session_capacity_source.json'
    spec = json.loads(spec_path.read_text())
    growth_path=package/'independent_session_growth_allocator_source.json'
    growth=json.loads(growth_path.read_text())
    patch=package/'independent_session_growth_allocator.patch'
    assert digest(patch)==growth['patch_SHA256']
    assert digest(spec_path)==growth['original_session_source_spec_SHA256']
    fast_path=package/'independent_session_stable_repartition_fastpath_source.json'
    fast=json.loads(fast_path.read_text()) if args.stable_fastpath else None
    fast_patch=package/'independent_session_stable_repartition_fastpath.patch'
    if fast:
        assert fast['allocator_growth_spec_SHA256']==digest(growth_path)
        assert fast['before_SHA256']==growth['after_SHA256'] and fast['file']==growth['file']
        assert digest(fast_patch)==fast['patch_SHA256']
    original_gate=json.loads(args.baseline_native_gate.read_text())
    assert original_gate['native_GPU_tests_pass'] and original_gate['test_count']==74
    assert Path(original_gate['source']).resolve()==args.baseline_source.resolve()
    assert original_gate['source_spec_SHA256']==digest(spec_path)
    for name,expected in original_gate['binaries'].items():assert digest(name)==expected
    if args.mini_preparation_job:
        original_mini=json.loads((args.mini_preparation_job/'progress.json').read_text())
        assert original_mini['status']=='mini-artifacts-ready'

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
                     source_spec_SHA256=digest(spec_path),allocator_growth_spec_SHA256=digest(growth_path),
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

    def verify(root,version):
        for name,record in spec['files'].items():
            expected=record['after_SHA256']
            if version=='after' and name==growth['file']:expected=growth['after_SHA256']
            if version=='after-fast' and name==growth['file']:expected=fast['after_SHA256']
            assert digest(root/name)==expected,name

    try:
        state('verifying-baseline-source')
        verify(args.baseline_source, 'before')
        controls = {}
        for name in ['backends/cuda/triton/kernels/int4_quantized_gemm.py',
                     'backends/cuda/triton/kernels/int6_quantized_gemm.py',
                     'backends/cuda/triton/kernels/quantized_gemm_utils.py',
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
        if fast:
            run(['git','apply','--check',fast_patch],cwd=source)
            run(['git','apply',fast_patch],cwd=source)
            verify(source,'after-fast')
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
             '--target', 'test_cuda_kv_cache'])
        test_binary = build / 'backends/cuda/test_cuda_kv_cache'
        test_binary_SHA256 = digest(test_binary)
        backend_lib = build / 'backends/cuda/libaoti_cuda_backend.a'
        backend_SHA256 = digest(backend_lib)
        run([cmake, '-S', source, '-B', build,
             '-DEXECUTORCH_BUILD_TESTS=OFF', '-DBUILD_TESTING=OFF'])
        run([cmake, '--build', build, '--parallel', args.jobs, '--target', 'install'])
        assert digest(test_binary) == test_binary_SHA256
        assert digest(backend_lib) == backend_SHA256, 'Native test backend changed during SDK install'
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
        verify(source, 'after-fast' if fast else 'after')
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
        assert len(tests)==74
        receipt = dict(native_GPU_tests_pass=True, test_count=len(tests),
                       independent_session_tests=independent,
                       source_spec_SHA256=digest(spec_path),allocator_growth_spec_SHA256=digest(growth_path), source=str(source),
                       source_spec_is_artifact_ABI_compatibility_reference=True,
                       actual_native_source_change=(dict(patches=[growth,fast]) if fast else growth),
                       stable_fastpath_spec_SHA256=digest(fast_path) if fast else None,
                       preserved_original_native_gate_SHA256=digest(args.baseline_native_gate),
                       binaries=binaries, unchanged_controls=controls,
                       native_tests_xml_SHA256=digest(xml),
                       model_numerics_pass=False, model_performance_pass=False,
                       full_model_export_authorized_by_this_receipt=False)
        receipt_path = args.output / 'native-gate.json'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        state('native-GPU-tests-passed', native_gate=str(receipt_path),
              test_count=len(tests), independent_session_tests=independent)
        if args.mini_preparation_job is None:
            state('native-GPU-tests-passed', native_gate=str(receipt_path),
                  native_preparation_only=True,
                  next_gate='Reuse independently qualified model artifacts; this receipt does not qualify a full model')
            return
        # Reuse the exact compiled mini graphs; only the native pool changes.
        alias=args.output/'same-artifact-mini-preparation';alias.mkdir()
        aliases=dict(receipt);aliases['binaries']=dict(receipt['binaries'])
        (alias/'model-build').mkdir()
        for name in ('muse_glimmer_runtime_kv_probe','muse_glimmer_batching_worker'):
            target=alias/'model-build'/name;target.symlink_to(model_build/name)
            aliases['binaries'][str(target)]=digest(model_build/name)
        for arm in ('global','independent'):
            original=args.mini_preparation_job/('mini-'+arm+'-artifact')
            audit=json.loads((original/'independent-mini-artifact-audit.json').read_text())
            for name,value in audit['artifacts'].items():assert digest(original/name)==value['sha256']
            (alias/('mini-'+arm+'-artifact')).symlink_to(original)
        (alias/'native-gate.json').write_text(json.dumps(aliases,indent=2)+'\n')
        (alias/'progress.json').write_text(json.dumps(dict(status='mini-artifacts-ready',
            controller_pid=os.getpid(),original_compiled_artifacts_unchanged=True,
            original_mini_preparation=str(args.mini_preparation_job),
            allocator_growth_spec_SHA256=digest(growth_path),
            full_model_export_started=False,candidate_accepted=False),indent=2)+'\n')
        state('native-GPU-tests-passed',native_gate=str(receipt_path),
            same_artifact_mini_preparation=str(alias),next_gate='same-PTE raw logits, cold/steady performance, actual graph replay, HTTP memory and reclaim')
    except Exception as error:
        state('failed', error_type=type(error).__name__, detail=str(error)[-2000:],
              traceback=traceback.format_exc()[-2500:])
        raise


if __name__ == '__main__':
    main()
