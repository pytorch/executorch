"""Prepare Gemma Python export sources after unchanged native ABI3 tests pass.

This CPU-only step preserves the tested SDK and applies Gemma's production
attention metadata changes. It neither exports a model nor promotes a runtime.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-preparation-job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-minutes', type=int, default=240)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    assert not args.output.exists(), 'Preserve prior source preparation'
    args.output.mkdir(parents=True)
    native_job = args.native_preparation_job / 'native-build'

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           GPU_work=False, model_exported=False, model_promoted=False,
                                           **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    def run(argv, cwd):
        with (args.output / 'preparation.log').open('a') as log:
            subprocess.run([str(v) for v in argv], cwd=cwd, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        deadline = time.monotonic() + 60 * args.wait_minutes
        while not (native_job / 'native-gate.json').is_file():
            progress = native_job / 'progress.json'
            if progress.is_file():
                current = json.loads(progress.read_text())
                assert current['status'] != 'failed', current
                pid = current['controller_pid']
            else:
                deployment = args.native_preparation_job / 'deployment.json'
                if deployment.is_file():
                    pid = json.loads(deployment.read_text())['native_controller_pid']
                else:
                    record = args.native_preparation_job.with_name(args.native_preparation_job.name + '-deployment.json')
                    pid = json.loads(record.read_text())['controller_pid']
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
            assert b'prepare_independent_session_' in command and str(args.native_preparation_job).encode() in command
            state('waiting-confirmed-live-native-preparation', native_controller_pid=pid)
            if time.monotonic() >= deadline:
                state('observation-timeout', native_controller_pid=pid, native_still_running=True)
                return
            time.sleep(20)
        gate_path = native_job / 'native-gate.json'
        gate = json.loads(gate_path.read_text())
        source = Path(gate['source'])
        shared = package / 'independent_session_capacity_source.json'
        assert digest(shared) == gate['source_spec_SHA256']
        spec = json.loads(shared.read_text())
        assert gate['native_GPU_tests_pass'] and gate['test_count'] == 74
        xml = native_job / 'native-tests.xml'
        assert digest(xml) == gate['native_tests_xml_SHA256']
        tests = list(ET.parse(xml).getroot().iter('testcase'))
        assert len(tests) == 74 and not any(t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None for t in tests)
        for name, record in spec['files'].items():
            assert digest(source / name) == record['after_SHA256'], name
        for name, expected in gate['unchanged_controls'].items():
            assert digest(source / name) == expected, name
        for name, expected in gate['binaries'].items():
            assert digest(name) == expected, name
        original = {name: digest(source / name) for name in spec['files']}
        target = args.output / 'executorch'
        state('copying-qualified-native-Python-source', native_gate_SHA256=digest(gate_path))

        def ignore(directory, names):
            return [name for name in names if name in ('.git', 'cmake-out', '__pycache__', '.pytest_cache') or
                    name.endswith(('.pte', '.ptd', '.gguf', '.log', '.nsys-rep'))]

        shutil.copytree(source, target, symlinks=True, ignore=ignore)
        baselines = {'model.py': '588ad608a7a69f9f213b1b72b05cdef871c105144c55dc12da0e972b80c65446',
                     'cuda_packers.py': 'c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e',
                     'gguf_loader.py': '936f2071aa1d93571da0aadc9d266a4d35ca4f2213e1ec0ca441788645c61c0d'}
        gemma = target / 'examples/models/gemma4_31b'
        for name, expected in baselines.items():
            assert digest(gemma / name) == expected, name
        for name in ('batching_model.py', 'batching_gguf_loader.py', 'cuda_source_transformations.py'):
            shutil.copy2(package / 'gemma_runtime_python' / name, gemma / name)
        export_path = gemma / 'export_batching.py'
        shutil.copy2(package / 'gemma_runtime_export_baseline.py', export_path)
        overlay = (package / 'gemma_runtime_columns_overlay.patch').read_text()
        export_diff = '--- a/examples/models/gemma4_31b/export_batching.py' + overlay.split(
            '--- a/examples/models/gemma4_31b/export_batching.py', 1)[1].split('--- a/extension/llm/batching/executor.h')[0]
        temporary = args.output / 'runtime-export-contract.patch'
        temporary.write_text(export_diff)
        run(['git', 'apply', '--check', temporary.resolve()], target)
        run(['git', 'apply', temporary.resolve()], target)
        exporter_spec = json.loads((package / 'gemma_independent_session_exporter.json').read_text())
        exporter_patch = package / 'gemma_independent_session_exporter.patch'
        assert digest(exporter_patch) == exporter_spec['patch_SHA256']
        for name, record in exporter_spec['files'].items():
            assert digest(target / name) == record['before_SHA256'], name
        run(['git', 'apply', '--check', exporter_patch], target)
        run(['git', 'apply', exporter_patch], target)
        for name, record in exporter_spec['files'].items():
            assert digest(target / name) == record['after_SHA256'], name
        attention_path = target / 'backends/cuda/triton/kernels/sdpa_varlen.py'
        baseline = package / 'gemma_runtime_attention_baseline.py'
        assert digest(baseline) == exporter_spec['production_attention_baseline_SHA256']
        shutil.copy2(baseline, attention_path)
        attention_spec = json.loads((package / 'gemma_independent_session_attention.json').read_text())
        patch = package / 'gemma_independent_session_attention.patch'
        assert digest(patch) == attention_spec['patch_SHA256']
        run(['git', 'apply', '--check', patch], target)
        run(['git', 'apply', patch], target)
        for name, record in attention_spec['files'].items():
            assert digest(target / name) == record['after_SHA256'], name
        for name, record in spec['files'].items():
            if name != 'backends/cuda/triton/kernels/sdpa_varlen.py':
                assert digest(target / name) == record['after_SHA256'], name
        for path in [export_path, attention_path, *(gemma / name for name in baselines),
                     *(gemma / name for name in ('batching_model.py', 'batching_gguf_loader.py', 'cuda_source_transformations.py'))]:
            ast.parse(path.read_text(), filename=str(path))
        assert original == {name: digest(source / name) for name in original}
        for name, expected in gate['binaries'].items():
            assert digest(name) == expected, name
        receipt = dict(original_native_gate_SHA256=digest(gate_path), original_native_gate=gate,
                       new_source=str(target), SDK=str(source / 'cmake-out'),
                       independent_exporter_spec_SHA256=digest(package / 'gemma_independent_session_exporter.json'),
                       production_attention_spec_SHA256=digest(package / 'gemma_independent_session_attention.json'),
                       source_files={str(path.relative_to(target)): digest(path) for path in
                                     [export_path, attention_path, *(gemma / name for name in baselines),
                                      *(gemma / name for name in ('batching_model.py', 'batching_gguf_loader.py', 'cuda_source_transformations.py'))]},
                       original_source_and_binaries_unchanged=True, native_tests_reused_with_identical_SDK=True,
                       actual_Gemma_model_graph_numerical_performance_validation_pending=True,
                       full_model_promoted=False, GPU_work=False)
        (args.output / 'source-preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        state('Gemma-independent-mini-source-ready', source=str(target), native_gate_SHA256=digest(gate_path))
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
