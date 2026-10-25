"""Build official-API drivers and measure seven fixed-1K-history batch mixes with only required histories.

No framework source patch, new kernel, activation precision change, shared
prefix reuse, or HTTP latency is introduced by these microbenchmarks.
Four fresh processes per framework; discard round one and average rounds2-4.
The parent has no CUDA context and owns the GPU lock through child teardown.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--et-source', type=Path, required=True)
    parser.add_argument('--et-link-provenance', type=Path, required=True)
    parser.add_argument('--cpp-source', type=Path, required=True)
    parser.add_argument('--cpp-build', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--export-job', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=20)
    args = parser.parse_args()
    args.job.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='4')

    def state(status, **fields):
        pending = args.job / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           **fields), indent=2) + '\n')
        pending.replace(args.job / 'progress.json')

    try:
        state('building-official-API-drivers')
        assert digest(args.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
        provenance = json.loads(args.et_link_provenance.read_text())
        # The previous official worker link command is immutable input. Do not
        # silently substitute a rebuilt runtime when reusing this SDK.
        assert isinstance(provenance['compile_command'], list)
        assert isinstance(provenance['link_command'], list)
        for path, expected in provenance['original_file_hashes'].items():
            assert digest(path) == expected, f'Preserved SDK/source input changed: {path}'
        actual_commands = []

        def command(cmd):
            actual_commands.append(cmd)
            with (args.job / 'build.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)

        obj = args.job / 'roofline_et.o'
        binary = args.job / 'roofline_et'
        compile_cmd = list(provenance['compile_command'])
        compile_cmd[compile_cmd.index('-c') + 1] = str(source / 'roofline_active_et.cpp')
        compile_cmd[compile_cmd.index('-o') + 1] = str(obj)
        compile_cmd[1:1] = ['-I' + str(args.et_source.parent)]
        command(compile_cmd)
        link_cmd = list(provenance['link_command'])
        old_obj = provenance['compile_command'][provenance['compile_command'].index('-o') + 1]
        assert link_cmd.count(old_obj) == 1
        link_cmd[link_cmd.index(old_obj)] = str(obj)
        link_cmd[link_cmd.index('-o') + 1] = str(binary)
        command(link_cmd)
        cpp_binary = args.job / 'roofline_cpp'
        lib = args.cpp_build / 'bin'
        command(['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++20', '-pthread',
                 '-I' + str(args.cpp_source / 'include'),
                 '-I' + str(args.cpp_source / 'ggml/include'),
                 '-I/usr/local/cuda/include', str(source / 'roofline_active_cpp.cpp'),
                 '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lllama',
                 '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
                 '-lcudart', '-o', str(cpp_binary)])
        cache = (args.cpp_build / 'CMakeCache.txt').read_text()
        assert 'GGML_CUDA:BOOL=ON' in cache
        assert 'GGML_CUDA_FORCE_CUBLAS:BOOL=ON' in cache
        inputs = dict(
            sources={name: digest(source / name) for name in
                     ['roofline_active_common.h', 'roofline_active_et.cpp', 'roofline_active_cpp.cpp', 'run_mg_roofline_active.py']},
            GGUF_sha256=digest(args.gguf),
            CPP_commit=subprocess.check_output(['git', '-C', str(args.cpp_source),
                                                'rev-parse', 'HEAD'], text=True).strip(),
            CPP_CMake_cache_sha256=digest(args.cpp_build / 'CMakeCache.txt'),
            ET_link_provenance_sha256=digest(args.et_link_provenance),
            preserved_SDK_and_source_hashes=provenance['original_file_hashes'],
            binaries={str(path): digest(path) for path in [binary, cpp_binary]},
            compile_and_link_commands=actual_commands,
            framework_sources_not_patched=True, new_KA_kernel_used=False,
            synthetic_input_definition='256+((owner+1)*7919+(position+1)*104729)%190000',
            histories_per_owner=1024, maximum_owners=32, history_population='only-current-mix; reset-between-cases', runtime_capacity_per_owner=2048)
        (args.job / 'build-and-input-provenance.json').write_text(json.dumps(inputs, indent=2) + '\n')
        state('waiting-independent-roofline-export')
        deadline = time.monotonic() + 21600
        while True:
            progress = json.loads((args.export_job / 'progress.json').read_text())
            assert progress['status'] != 'failed', 'Independent export failed; do not run a different artifact'
            if progress['status'] == 'completed':
                break
            assert time.monotonic() < deadline, 'Export remains incomplete; retain attempts'
            time.sleep(30)
        assert (args.artifact / 'model.pte').is_file()
        assert (args.artifact / 'aoti_cuda_blob.ptd').is_file()
        inputs['artifact_sha256'] = {name: digest(args.artifact / name) for name in
                                    ['model.pte', 'aoti_cuda_blob.ptd']}
        (args.job / 'build-and-input-provenance.json').write_text(json.dumps(inputs, indent=2) + '\n')
        all_results = {'ET': [], 'CPP': []}
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip(), \
                'Existing GPU work must remain untouched'
            for round_id in range(1, 5):
                order = ['ET', 'CPP'] if round_id % 2 else ['CPP', 'ET']
                for framework in order:
                    state('measuring', round=round_id, framework=framework, first_round_discarded=True)
                    result = args.job / f'{framework}-round{round_id}.json'
                    if framework == 'ET':
                        cmd = [str(binary), str(args.artifact / 'model.pte'),
                               str(args.artifact / 'aoti_cuda_blob.ptd'), str(result),
                               str(args.repetitions)]
                    else:
                        cmd = [str(cpp_binary), str(args.gguf), str(result), str(args.repetitions)]
                    with (args.job / f'{framework}-round{round_id}.log').open('x') as log:
                        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=3600)
                    data = json.loads(result.read_text())
                    assert data['completed'] and len(data['cases']) == 7
                    assert all(case['same_prefix_greedy_repeats_equal'] for case in data['cases'])
                    assert data['history_population']=='only-current-mix; reset-between-cases'
                    assert all(case['populated_private_histories']==case['decode_rows']+bool(case['prefill_rows']) for case in data['cases'])
                    all_results[framework].append(data)
                    deadline = time.monotonic() + 60
                    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                    '--format=csv,noheader'], text=True).strip():
                        assert time.monotonic() < deadline, 'Wait for owned CUDA context teardown'
                        time.sleep(1)
        results = []
        for framework, rounds in all_results.items():
            for index in range(7):
                cases = [row['cases'][index] for row in rounds]
                assert len({row['mix'] for row in cases}) == 1
                round_ms = [statistics.mean(row['latency_ms']) for row in cases]
                round_tps = [cases[0]['batch_tokens'] * 1000 / ms for ms in round_ms]
                results.append(dict(framework=framework, mix=cases[0]['mix'],
                                    batch_tokens=cases[0]['batch_tokens'],
                                    latency_ms_rounds=round_ms, processed_TPS_rounds=round_tps,
                                    latency_ms_mean=statistics.mean(round_ms[1:]),
                                    latency_ms_sample_SD=statistics.stdev(round_ms[1:]),
                                    processed_TPS_mean=statistics.mean(round_tps[1:]),
                                    processed_TPS_sample_SD=statistics.stdev(round_tps[1:]),
                                    first_round_discarded=True))
        (args.job / 'results.json').write_text(json.dumps(dict(
            rounds=4, discarded=[1], averaged=[2, 3, 4], results=results,
            comparison_scope='official API forward + device greedy, only the requested private 1K histories per mix',
            not_HTTP_output_throughput=True,
            actual_kernel_time_or_DRAM_counters_measured=False,
            artifact_server_quality_not_replaced=True), indent=2) + '\n')
        state('completed', results=14, deployment_accepted=False,
              independent_numerical_and_trace_review_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1600:])
        raise


if __name__ == '__main__':
    main()
