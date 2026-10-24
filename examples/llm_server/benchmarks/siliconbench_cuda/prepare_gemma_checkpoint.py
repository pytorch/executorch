"""Expose the shared ET prefix checkpoint policy in an isolated Gemma worker.

Reuse the previously audited graph-enabled worker and its native libraries.
No checkout, installed launcher or baseline binary is changed. Preparation
alone is not model correctness or performance acceptance.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--et-source', type=Path, required=True)
    parser.add_argument('--graph-worker-source', type=Path, required=True)
    parser.add_argument('--build-inputs', type=Path, required=True)
    parser.add_argument('--base-launcher', type=Path, required=True)
    parser.add_argument('--recorded-et-source', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing builds and results'
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          model_acceptance_pending=True, **fields), indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    try:
        state('waiting-GPU-lock')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
            inputs = args.build_inputs.resolve()
            et = args.et_source.resolve()
            worker = args.graph_worker_source.resolve()
            launcher = args.base_launcher.resolve()
            header = inputs/'include/executorch/extension/llm/serving/serving_runtime.h'
            serving_lib = inputs/'lib/libextension_llm_serving.a'
            build_audit = inputs/'build-audit.json'
            assert digest(worker) == '03504329228e76c27d0f4515f69a6528d2214c6434909f1ac443500acde884b2'
            assert digest(launcher) == '83d55dd793cc7eeffb0debc7e6ac13db782795c90132055470804b3a09bbab57'
            assert digest(header) == '91ee3da837addd57da8627cd3965b015199e7dbfd316f064948ad963e19cff99'
            assert digest(serving_lib) == 'ccefde5d4ecbb23afdab2975832b2e151c06d5310cc39afffd623b9919d83e87'
            assert digest(build_audit) == '39aae0c8199d6f9e2f3da929ee61f66a1d8de6386e7e3ced6e7c697ddce9e39e'
            tracked = [worker,launcher,header,build_audit,*sorted((inputs/'lib').glob('*'))]
            before = {str(p):digest(p) for p in tracked if p.is_file()}
            text = worker.read_text()
            text = replace_once(text, 'DEFINE_int32(prefix_cache_entries, 16, "Prefix snapshots");',
                                'DEFINE_int32(prefix_cache_entries, 16, "Prefix snapshots");\n'
                                'DEFINE_int32(prefix_checkpoint_interval, 0, "Periodic prefix checkpoint tokens; zero disables periodic capture");')
            text = replace_once(text, 'FLAGS_prefix_cache_entries < 0 || physical >',
                                'FLAGS_prefix_cache_entries < 0 || FLAGS_prefix_checkpoint_interval < 0 ||\n'
                                '        (FLAGS_prefix_checkpoint_interval > 0 && FLAGS_prefix_cache_entries == 0) || physical >')
            text = replace_once(text, 'config.prefix_cache_capacity = FLAGS_prefix_cache_entries;',
                                'config.prefix_cache_capacity = FLAGS_prefix_cache_entries;\n'
                                '    config.prefix_checkpoint_interval = FLAGS_prefix_checkpoint_interval;')
            source = args.output/'gemma4_checkpoint_worker.cpp'
            source.write_text(text)
            py = launcher.read_text()
            py = replace_once(py, 'f"--prefix_cache_entries={args.prefix_cache_entries}"]',
                              'f"--prefix_cache_entries={args.prefix_cache_entries}",\n'
                              '        f"--prefix_checkpoint_interval={args.prefix_checkpoint_interval}"]')
            py = replace_once(py, 'parser.add_argument("--prefix-cache-entries", type=int, default=16)',
                              'parser.add_argument("--prefix-cache-entries", type=int, default=16)\n'
                              '    parser.add_argument("--prefix-checkpoint-interval", type=int, default=0)')
            py = replace_once(py, 'args.max_inflight_requests) <= 0 or args.prefix_cache_entries < 0:',
                              'args.max_inflight_requests) <= 0 or args.prefix_cache_entries < 0 or \\\n'
                              '           args.prefix_checkpoint_interval < 0 or \\\n'
                              '           (args.prefix_checkpoint_interval > 0 and args.prefix_cache_entries == 0):')
            ast.parse(py)
            generated_launcher = args.output/'serve_gemma_checkpoint.py'
            generated_launcher.write_text(py)
            audit = json.loads(build_audit.read_text())
            recorded = str(args.recorded_et_source)
            compile_command = audit['compile_command'].copy()
            for i, arg in enumerate(compile_command):
                compile_command[i] = arg.replace(recorded+'/cmake-out/include',str(inputs/'include')).replace(
                    recorded+'/cmake-out/third-party/gflags/include',str(inputs/'gflags-include')).replace(recorded,str(et))
            obj = args.output/'worker.o'
            compile_command[compile_command.index('-c')+1] = str(source.resolve())
            compile_command[compile_command.index('-o')+1] = str(obj.resolve())
            link_command = audit['link_command'].copy()
            assert sum(arg.endswith('.o') for arg in link_command) == 1
            for i, arg in enumerate(link_command):
                if arg.endswith('.o'):
                    link_command[i] = str(obj.resolve())
                elif arg.startswith('lib') and arg.endswith('.a'):
                    link_command[i] = str(inputs/'lib'/arg)
                elif recorded+'/cmake-out/lib/' in arg or recorded+'/cmake-out/third-party/gflags/libgflags_' in arg:
                    start = arg.index(recorded)
                    end = arg.find(',',start)
                    if end < 0:
                        end = len(arg)
                    link_command[i] = arg[:start]+str(inputs/'lib'/Path(arg[start:end]).name)+arg[end:]
                elif arg.startswith('-Wl,-rpath,'):
                    link_command[i] = '-Wl,-rpath,'+str(inputs/'lib')+':/usr/local/cuda/lib64'
                elif arg.startswith('/usr/local/cuda-13.2/') and not Path(arg).exists():
                    link_command[i] = arg.replace('/usr/local/cuda-13.2/','/usr/local/cuda/')
            binary = args.output/'gemma4_checkpoint_worker'
            link_command[link_command.index('-o')+1] = str(binary.resolve())
            state('building-isolated-worker')
            with (args.output/'build.log').open('x') as log:
                subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                subprocess.run(link_command, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            help_result = subprocess.run([str(binary.resolve()),'--helpshort'], capture_output=True, text=True, timeout=30)
            assert 'prefix_checkpoint_interval' in help_result.stdout+help_result.stderr
            # CLI help exercises imports and argument construction without
            # loading a model or starting a GPU worker.
            help_result = subprocess.run([args.python,'-B',str(generated_launcher.resolve()),'--help'],
                                         capture_output=True, text=True, check=True, timeout=60)
            assert '--prefix-checkpoint-interval' in help_result.stdout
            assert before == {str(p):digest(p) for p in tracked if p.is_file()}
            record = dict(worker_binary=str(binary.resolve()), launcher=str(generated_launcher.resolve()),
                          binary_sha256=digest(binary), worker_source_sha256=digest(source),
                          launcher_sha256=digest(generated_launcher), original_source_and_libraries_unchanged=True,
                          original_file_hashes=before, compile_command=compile_command, link_command=link_command,
                          static_graphs_enabled_by_default=True, existing_shared_checkpoint_runtime=True,
                          original_raw_streaming_and_tool_text_preserved=True,
                          no_model_loaded_during_preparation=True, model_correctness_perf_memory_acceptance_pending=True)
            (args.output/'build-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        state('completed', isolated_worker_prepared=True, model_acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
