"""Wait for task-owned preparation, then collect the stock agent matrix.

Observe the existing preparation service without restarting it. Keep results
outside the source checkout and preserve failed attempts. ET selection and final
quality, memory and cache acceptance remain separate from stock data collection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-output', type=Path, required=True)
    parser.add_argument('--preparation-program', required=True)
    parser.add_argument('--data-revision', required=True)
    parser.add_argument('--base-wrapper', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--native-inputs', type=Path)
    parser.add_argument('--python', required=True)
    parser.add_argument('--cpp-off-build', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19082)
    parser.add_argument('--max-context', type=int, choices=[12288,131072], default=12288)
    parser.add_argument('--initial-kv-capacity', type=int, default=512)
    parser.add_argument('--cpp-pool-tokens', type=int)
    args = parser.parse_args()
    assert all(c.isalnum() or c in '-_' for c in args.preparation_program)
    assert len(args.data_revision) == 40
    assert not args.output.exists(), 'Preserve previous attempts'
    assert args.base_wrapper.is_file() and args.client.is_file() and args.prompts.is_file()
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    try:
        state('waiting-preparation', preparation_program=args.preparation_program)
        while True:
            path = args.preparation_output / 'progress.json'
            preparation = json.loads(path.read_text()) if path.exists() else None
            if preparation and preparation['status'] == 'completed':
                break
            if preparation:
                assert preparation['status'] not in ['failed', 'cancelled', 'aborted'], preparation
            service = subprocess.run(['supervisorctl', 'status', args.preparation_program],
                                     capture_output=True, text=True, timeout=30)
            assert service.returncode == 0 and any(value in service.stdout.split()
                                                  for value in ['RUNNING', 'STARTING']), service.stdout.strip()
            time.sleep(30)
        assert preparation['stock_cpp_off_build_completed'] and preparation['correctness_inputs_pinned']
        data = json.loads((args.preparation_output / 'GMRID-input-audit.json').read_text())
        build = json.loads((args.preparation_output / 'stock-cpp-off-build.json').read_text())
        assert data['revision'] == args.data_revision and data['evaluated_items'] == 1146
        assert build['no_framework_code_patches'] and build['stock_tracked_source_clean']
        assert build['changed_requested_option'] == {'GGML_CUDA_FORCE_CUBLAS': ['ON', 'OFF']}
        assert Path(build['binary']).parents[1].name == args.cpp_off_build
        source = Path(__file__).resolve().parent
        wrapper = args.output / 'serve-public-wrapper.sh'
        subprocess.run([sys.executable, '-B', str(source / 'make_wrapper.py'),
                        '--base', str(args.base_wrapper), '--output', str(wrapper)], check=True)
        if args.max_context != 12288:
            from configure_context import configure
            wrapper.write_text(configure(wrapper.read_text(), args.max_context,
                                         args.initial_kv_capacity, args.cpp_pool_tokens or args.max_context))
            subprocess.run(['bash', '-n', str(wrapper)], check=True)
        matrix = args.output / 'matrix'
        command = [args.python, '-B', str(source / 'run_matrix.py'), '--wrapper', str(wrapper),
                   '--model', args.model, '--client', str(args.client), '--prompts', str(args.prompts),
                   '--correctness-root', str(args.preparation_output / 'correctness'),
                   '--python', args.python, '--cpp-off-build', args.cpp_off_build,
                   '--lock', str(args.lock), '--output', str(matrix), '--port', str(args.port),
                   '--variants', 'llamacpp-off', 'vllm', 'sglang', 'llamacpp-on',
                   '--cache-modes', 'off', 'on', '--concurrencies', '16', '8', '4', '2', '1']
        if args.native_inputs:
            command += ['--native-inputs', str(args.native_inputs)]
        with (args.output / 'matrix-controller.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state('running-matrix', matrix_controller_pid=child.pid, total_cases=40,
                  ET_best_configuration_acceptance_pending=True)
            assert child.wait() == 0, 'Inspect the preserved matrix/controller logs'
        assert json.loads((matrix / 'progress.json').read_text())['status'] == 'completed'
        state('completed', total_cases=40, performance_correctness_memory_collected=True,
              final_quality_cache_memory_and_best_ET_acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
