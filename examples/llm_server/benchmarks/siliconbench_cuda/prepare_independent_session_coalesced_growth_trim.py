"""Build coalesced allocator growth trims in an isolated verified ABI3 SDK copy.

Preserve exact PTE/PTD, model kernels, sliding storage and every baseline.
Trim unused caches before/after genuine growth; retain contraction behavior.
This experimental policy requires native, mini correctness and performance
checks before any full-model runtime deployment or main acceptance.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'baseline-native-job', 'python', 'output', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-variant', choices=('MG-empty-owner-reuse', 'Gemma-empty-owner-reuse', 'growth-only'), required=True)
    parser.add_argument('--allocator-policy', choices=('coalesced', 'pool-only', 'workspace-preserving', 'isolated-KV', 'isolated-KV-growth-slack'), default='coalesced')
    parser.add_argument('--mini-preparation-job', type=Path)
    parser.add_argument('--mini-profile', choices=('muse-glimmer-first4', 'gemma4-first6'), required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    assert args.package.is_dir() and args.python.is_file()
    sys.path.insert(0, str(args.package.resolve()))
    binding = importlib.import_module('independent_session_full_native_binding')
    profiles = importlib.import_module('independent_session_mini_profiles')
    baseline = binding.verify_runtime(args.baseline_native_job, args.package)
    source_before = Path(baseline['source'])
    assert source_before.is_dir()
    resources = Path(__file__).resolve().parent
    policy = 'independent_session_pool_only_growth_trim' if args.allocator_policy == 'pool-only' else 'independent_session_coalesced_growth_trim'
    policy_key = 'pool_only_growth_trim_spec_SHA256' if args.allocator_policy == 'pool-only' else 'coalesced_growth_trim_spec_SHA256'
    if args.allocator_policy == 'workspace-preserving':
        policy = 'independent_session_workspace_preserving_growth_trim'
        policy_key = 'workspace_preserving_growth_trim_spec_SHA256'
    if args.allocator_policy == 'isolated-KV':
        policy = 'independent_session_isolated_KV_pool'
        policy_key = 'isolated_KV_pool_spec_SHA256'
    if args.allocator_policy == 'isolated-KV-growth-slack':
        policy = 'independent_session_isolated_KV_growth_slack'
        policy_key = 'isolated_KV_growth_slack_spec_SHA256'
    fix_path = resources / (policy + '_source.json')
    fix = json.loads(fix_path.read_text())
    patch = resources / (policy + '.patch')
    assert sha(patch) == fix['patch_SHA256']
    records = fix['variants'][args.source_variant]
    for record in records:
        file = source_before / record['file']
        assert not file.is_symlink() and sha(file) == record['before_SHA256'], record['file']
    assert not args.output.exists(), 'Preserve previous attempts'
    args.output.mkdir(parents=True)
    source = args.output / 'executorch'
    state_path = args.output / 'progress.json'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'CUDAARCHS': '120',
           'TORCH_CUDA_ARCH_LIST': '12.0', 'CUDA_HOME': '/usr/local/cuda',
           'PYTHONDONTWRITEBYTECODE': '1', 'MAX_JOBS': str(args.jobs),
           'OMP_NUM_THREADS': str(args.jobs), 'PYTHONPATH': str(source.parent)}
    env['PATH'] = str(args.python.parent) + os.pathsep + env['PATH']

    def state(status, **extra):
        doc = {'status': status, 'controller_pid': os.getpid(),
               'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source': str(source), 'source_spec_SHA256': baseline['source_spec_SHA256'],
               policy_key: sha(fix_path),
               'baseline_modified': False, 'candidate_accepted': False,
               'full_model_export_started': False, **extra}
        pending = state_path.with_suffix('.pending.json')
        pending.write_text(json.dumps(doc, indent=2) + '\n')
        pending.replace(state_path)
        with (args.output / 'history.jsonl').open('a') as log:
            log.write(json.dumps(doc) + '\n')

    def run(argv, *, run_env=None, log_name='build.log', timeout=14400, cwd=None):
        argv = [str(v) for v in argv]
        with (args.output / log_name).open('a') as log:
            log.write(json.dumps({'argv': argv, 'cwd': str(cwd)}) + '\n')
            log.flush()
            subprocess.run(argv, env=run_env or env, cwd=cwd, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def verify_preserved_baseline():
        verified = binding.verify_runtime(args.baseline_native_job, args.package)
        assert verified == baseline

    try:
        state('copying-qualified-native-source')
        def ignore(directory, names):
            return [name for name in names if name in (
                '.git', 'cmake-out', '__pycache__', '.pytest_cache')
                or name.endswith(('.pte', '.ptd', '.gguf', '.log', '.nsys-rep'))]
        shutil.copytree(source_before, source, symlinks=True, ignore=ignore)
        spec = json.loads((args.package / 'independent_session_capacity_source.json').read_text())
        controls = dict(baseline['unchanged_controls'])
        before_hashes = {name: sha(source_before / name) for name in spec['files']}
        before_hashes.update({name: sha(source_before / name) for name in controls})
        assert all(sha(source / name) == expected for name, expected in before_hashes.items())
        run(['git', 'apply', '--check', patch], cwd=source)
        run(['git', 'apply', patch], cwd=source)
        changed = {record['file']: record for record in records}
        for name, expected in before_hashes.items():
            after = changed[name]['after_SHA256'] if name in changed else expected
            assert sha(source / name) == after, name
        verify_preserved_baseline()
        cmake = shutil.which('cmake', path=env['PATH'])
        assert cmake
        prefix = subprocess.check_output([args.python, '-B', '-c',
            'import torch; print(torch.utils.cmake_prefix_path)'], env=env, text=True).strip()
        build = source / 'cmake-out'
        state('building-native-core')
        run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DEXECUTORCH_BUILD_PRESET_FILE=' + str(source / 'tools/cmake/preset/llm.cmake'),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DEXECUTORCH_BUILD_PYBIND=OFF',
             '-DEXECUTORCH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON',
             '-DCMAKE_INSTALL_PREFIX=' + str(build), '-DCMAKE_PREFIX_PATH=' + prefix,
             '-DPYTHON_EXECUTABLE=' + str(args.python)])
        run([cmake, '--build', build, '--parallel', args.jobs, '--target', 'test_cuda_kv_cache'])
        test_binary = build / 'backends/cuda/test_cuda_kv_cache'
        backend = build / 'backends/cuda/libaoti_cuda_backend.a'
        sealed = {test_binary: sha(test_binary), backend: sha(backend)}
        run([cmake, '-S', source, '-B', build, '-DEXECUTORCH_BUILD_TESTS=OFF', '-DBUILD_TESTING=OFF'])
        run([cmake, '--build', build, '--parallel', args.jobs, '--target', 'install'])
        assert all(sha(p) == expected for p, expected in sealed.items())
        model_build = args.output / 'model-build'
        state('building-same-ABI-mini-probe')
        run([cmake, '-S', source / 'examples/models/muse-glimmer', '-B', model_build,
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + str(build),
             '-DEXECUTORCH_BUILD_CUDA=ON', '-DMUSE_GLIMMER_BUILD_BATCHING=ON',
             '-DMUSE_GLIMMER_BUILD_RUNTIME_KV_PROBE=ON'])
        run([cmake, '--build', model_build, '--parallel', args.jobs, '--target',
             'muse_glimmer_runtime_kv_probe', 'muse_glimmer_batching_worker'])
        binaries = {str(p): sha(p) for p in (test_binary,
            model_build / 'muse_glimmer_runtime_kv_probe', model_build / 'muse_glimmer_batching_worker')}
        xml = args.output / 'native-tests.xml'
        state('waiting-native-GPU-tests', binaries=binaries)
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip()
            state('running-native-GPU-tests', binaries=binaries)
            run([test_binary, '--gtest_output=xml:' + str(xml)],
                run_env={**env, 'CUDA_VISIBLE_DEVICES': '0'}, log_name='native-tests.log', timeout=1800)
        tests = list(ET.parse(xml).getroot().iter('testcase'))
        assert len(tests) == 74
        assert not any(t.find('failure') is not None or t.find('error') is not None or
                       t.find('skipped') is not None for t in tests)
        independent = [t.attrib['name'] for t in tests
                       if t.attrib['classname'] == 'CudaIndependentSessionCacheTest']
        assert len(independent) == 7 and set(independent) == set(baseline['independent_session_tests'])
        assert all(sha(p) == expected for p, expected in sealed.items())
        prior = baseline['actual_native_source_change']
        prior_records = prior['patches'] if 'patches' in prior else [prior]
        receipt = {**baseline, 'source': str(source), 'binaries': binaries,
                   'native_tests_xml_SHA256': sha(xml), 'independent_session_tests': independent,
                   'preserved_preceding_native_gate_SHA256': sha(args.baseline_native_job / 'native-gate.json'),
                   'actual_native_source_change': {'patches': prior_records + records},
                   policy_key: sha(fix_path),
                   'model_numerics_pass': False, 'model_performance_pass': False,
                   'full_model_export_authorized_by_this_receipt': False}
        (args.output / 'native-gate.json').write_text(json.dumps(receipt, indent=2) + '\n')
        binding.verify_runtime(args.output, args.package)
        verify_preserved_baseline()
        if args.mini_preparation_job:
            deadline = time.monotonic() + 4 * 3600
            while True:
                mini_state = json.loads((args.mini_preparation_job / 'progress.json').read_text())
                if mini_state['status'] == 'mini-artifacts-ready':
                    break
                assert mini_state['status'] != 'failed', mini_state
                pid = mini_state['controller_pid']
                command = Path('/proc') / str(pid) / 'cmdline'
                assert command.is_file(), 'Preserve terminal mini export for review'
                actual_command = command.read_bytes()
                assert b'export_independent_session_mini_from_native.py' in actual_command
                assert str(args.mini_preparation_job).encode() in actual_command
                assert time.monotonic() < deadline, 'Confirmed-live mini export observation expired'
                state('waiting-on-confirmed-live-mini-artifacts', mini_PID=pid,
                      mini_status=mini_state['status'], native_tests_passed=True)
                time.sleep(20)
            alias = args.output / 'same-artifact-mini-preparation'
            alias.mkdir()
            aliases = dict(receipt)
            aliases['binaries'] = dict(binaries)
            (alias / 'model-build').mkdir()
            for name in ('muse_glimmer_runtime_kv_probe', 'muse_glimmer_batching_worker'):
                target = alias / 'model-build' / name
                target.symlink_to(model_build / name)
                aliases['binaries'][str(target)] = sha(model_build / name)
            for arm in ('global', 'independent'):
                original = args.mini_preparation_job / ('mini-' + arm + '-artifact')
                audit = json.loads((original / 'independent-mini-artifact-audit.json').read_text())
                profiles.validate_mini_audit(audit, args.mini_profile, arm == 'independent')
                assert audit['source_spec_sha256'] == receipt['source_spec_SHA256']
                for name, expected in audit['artifacts'].items():
                    assert sha(original / name) == expected['sha256']
                (alias / ('mini-' + arm + '-artifact')).symlink_to(original)
            (alias / 'native-gate.json').write_text(json.dumps(aliases, indent=2) + '\n')
            (alias / 'progress.json').write_text(json.dumps({
                'status': 'mini-artifacts-ready', 'controller_pid': os.getpid(),
                'original_compiled_artifacts_unchanged': True, 'candidate_accepted': False,
                'full_model_export_started': False}, indent=2) + '\n')
        state('native-GPU-tests-passed', native_gate=str(args.output / 'native-gate.json'),
              test_count=74, independent_session_tests=independent,
              next_gate='same-PTE native ABBA, raw numerics, cold/steady performance, capacity/rewind/graphs and HTTP lifecycle')
    except Exception as error:
        state('failed', error_type=type(error).__name__, detail=str(error)[-2000:],
              traceback=traceback.format_exc()[-2500:])
        raise


if __name__ == '__main__':
    main()
