"""One CUDA SiliconBench server configuration, memory sampling and GMRID F1.

Framework/model loading is supplied by an external launch wrapper. This driver
changes no framework source or kernel. Results stay under the caller's output.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--backend', choices=['executorch', 'vllm', 'sglang', 'llamacpp'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--prefix-cache', choices=['on', 'off'], required=True)
    parser.add_argument('--cpp-build', default='build-stock-cublas')
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--native-inputs', type=Path)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', default='python3')
    parser.add_argument('--concurrency', type=int, choices=[1, 2, 4, 8, 16], required=True)
    parser.add_argument('--port', type=int, default=19082)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--correctness-rounds', type=int, default=4)
    args = parser.parse_args()
    assert args.correctness_rounds > 0
    data_audit_path = args.correctness_root.parent/'GMRID-input-audit.json'
    data_audit = json.loads(data_audit_path.read_text())
    expected_items = data_audit['evaluated_items']
    assert expected_items == 1146 and data_audit['source_rows'] == 1147
    assert data_audit['excluded_source_rows'] == [dict(id='5079', reasons=['empty_input'], gold_label='Others')]
    for shots in [0, 5]:
        for name in ['prompts', 'labels']:
            path = args.correctness_root/f'prompts/{shots}shot/{name}.jsonl'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == data_audit['prompt_sets'][str(shots)][name+'_sha256']
    assert not args.output.exists(), 'Preserve incomplete cases; use a distinct output for a reviewed retry'
    args.output.mkdir(parents=True)
    started = time.monotonic()
    server = None
    stop_sampler = threading.Event()
    memory = []
    phase = 'baseline'

    def progress(status, **fields):
        record = dict(status=status, controller_pid=os.getpid(), elapsed_s=time.monotonic()-started, **fields)
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(record, indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    def gpu_apps():
        return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()

    def sample_memory():
        while not stop_sampler.is_set():
            try:
                row = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,power.draw,clocks.sm',
                                               '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                values = [float(x.strip()) for x in row.split(',')]
                processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,rss'], text=True, timeout=15)
                table = [tuple(int(x) for x in line.split()) for line in processes.splitlines()[1:]]
                owners = {server.pid} if server is not None else set()
                while True:
                    children = {pid for pid, parent, _ in table if parent in owners}
                    if children.issubset(owners):
                        break
                    owners.update(children)
                rss = sum(value for pid, _, value in table if pid in owners)
                memory.append(dict(elapsed_s=time.monotonic()-started, phase=phase,
                                   GPU_used_MiB=values[0], power_W=values[1], clocks_SM_MHz=values[2],
                                   server_tree_host_RSS_KiB=rss))
            except (ValueError, subprocess.SubprocessError) as exc:
                memory.append(dict(phase=phase, error_type=type(exc).__name__))
            stop_sampler.wait(1)

    def run(command, log, timeout):
        with log.open('x') as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    lock = args.lock.open('a')
    try:
        progress('waiting-GPU-lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not gpu_apps(), 'Existing GPU work is not ours; leave it untouched'
        env = dict(os.environ, SB_PORT=str(args.port), SB_PREFIX_CACHE=args.prefix_cache,
                   SB_CPP_BUILD=args.cpp_build, APPLEBENCH_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        if args.native_inputs and args.backend == 'sglang':
            env['SB_NATIVE_COMPLETION_PROMPTS'] = str(args.native_inputs)
        else:
            env.pop('SB_NATIVE_COMPLETION_PROMPTS', None)
        settings = dict(model=args.model, backend=args.backend, variant=args.variant,
                        concurrency=args.concurrency, prefix_cache=args.prefix_cache == 'on',
                        performance_rounds=4, discarded=[1], averaged=[2,3,4], warmup_each_round=3,
                        correctness_rounds=args.correctness_rounds, correctness_items=expected_items, shots=[0,5],
                        source_rows=data_audit['source_rows'], excluded_source_rows=data_audit['excluded_source_rows'],
                        correctness_input_audit_sha256=hashlib.sha256(data_audit_path.read_bytes()).hexdigest(),
                        wrapper_sha256=hashlib.sha256(args.wrapper.read_bytes()).hexdigest(),
                        client_sha256=hashlib.sha256(args.client.read_bytes()).hexdigest(),
                        prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                        GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit',
                                                     '--format=csv,noheader'], text=True).strip(),
                        source_code_patches_by_driver=False,
                        metric='server output tokens / complete HTTP workload wall time',
                        memory='sampled total device memory and server process-tree RSS; not allocator allocated/reserved')
        (args.output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        stop_sampler.wait(2)
        phase = 'loading'
        log = (args.output/'server.log').open('x')
        server = subprocess.Popen(['bash', str(args.wrapper), args.backend], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        progress('starting-server', server_pid=server.pid)
        deadline = time.monotonic()+2400
        while True:
            assert server.poll() is None, 'Server exited; inspect its remote log'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as response:
                    if json.load(response).get('data'):
                        break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Server did not become ready')
            stop_sampler.wait(5)
        phase = 'ready'
        stop_sampler.wait(2)
        rounds = []
        for repetition in [1,2,3,4]:
            phase = 'performance'
            progress('running-performance', round=repetition, server_pid=server.pid)
            target = args.output/f'repeat-{repetition}'
            target.mkdir()
            run([args.python, '-B', str(args.client), '--port', str(args.port), '--framework', args.backend,
                 '--split', 'agent', '--concurrency', str(args.concurrency), '--requests', '100',
                 '--warmup', '3', '--prompts', str(args.prompts), '--output', str(target/'metrics.json'),
                 '--outputs', str(target/'outputs.jsonl')], target/'client.log', 7200)
            metrics = json.loads((target/'metrics.json').read_text())['concurrency_results'][0]
            assert metrics['successful']+metrics['failed'] == 100
            assert abs(metrics['total_output_tokens']/metrics['wall_time_s']-metrics['output_throughput_tps']) < 1e-6
            responses = [json.loads(line) for line in (target/'outputs.jsonl').open()]
            assert len(responses) == 100 and len({row['request_idx'] for row in responses}) == 100
            rounds.append(metrics)
        keys = ['output_throughput_tps', 'wall_time_s', 'total_output_tokens', 'total_input_tokens',
                'ttft_avg_ms', 'latency_avg_s', 'successful', 'failed']
        aggregate = {key:dict(mean=statistics.mean(row[key] for row in rounds[1:]),
                              sample_stddev=statistics.stdev(row[key] for row in rounds[1:]),
                              first_discarded=rounds[0][key], values=[row[key] for row in rounds[1:]])
                     for key in keys if key in rounds[0]}
        (args.output/'performance-aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n')
        phase = 'post-performance-idle'
        stop_sampler.wait(2)
        scores = []
        for repetition in range(1, args.correctness_rounds+1):
            for shots in [0,5]:
                phase = 'correctness'
                progress('running-correctness', round=repetition, shots=shots, server_pid=server.pid)
                target = args.output/f'correctness/round-{repetition}/{shots}shot'
                target.mkdir(parents=True)
                inputs = args.correctness_root/f'prompts/{shots}shot'
                prompts = [json.loads(line) for line in (inputs/'prompts.jsonl').open()]
                assert len(prompts) == expected_items
                provenance = dict(settings, correctness_round=repetition, shots=shots,
                                  correctness_prompts_sha256=hashlib.sha256((inputs/'prompts.jsonl').read_bytes()).hexdigest())
                run([args.python, '-B', str(args.correctness_root/'scripts/run_eval.py'),
                     '--base-url', f'http://127.0.0.1:{args.port}', '--model', args.model,
                     '--prompts', str(inputs/'prompts.jsonl'), '--output', str(target/'responses.jsonl'),
                     '--concurrency', str(args.concurrency), '--max-retries', '0'], target/'evaluation.log', 21600)
                responses = [json.loads(line) for line in (target/'responses.jsonl').open()]
                assert len(responses) == expected_items
                assert len({str(row['id']) for row in responses}) == expected_items
                assert {str(row['id']) for row in responses} == {str(row['id']) for row in prompts}
                run([args.python, '-B', str(args.correctness_root/'scripts/score_f1.py'),
                     '--responses', str(target/'responses.jsonl'), '--labels', str(inputs/'labels.jsonl'),
                     '--categories', str(args.correctness_root/'data/categories.json'),
                     '--output', str(target/'scores.json'), '--provenance', json.dumps(provenance)],
                    target/'scoring.log', 180)
                score = json.loads((target/'scores.json').read_text())
                assert score['n_scored'] == expected_items and score['n_missing_responses'] == score['n_extra_responses'] == 0
                scores.append(dict(round=repetition, shots=shots, **score))
        (args.output/'correctness-scores.json').write_text(json.dumps(scores, indent=2)+'\n')
        correctness_aggregate = {}
        for shots in [0,5]:
            rows = [row for row in scores if row['shots'] == shots]
            measured = rows[1:] if len(rows) > 1 else rows
            correctness_aggregate[str(shots)] = {
                key: dict(mean=statistics.mean(row[key] for row in measured),
                          sample_stddev=statistics.stdev(row[key] for row in measured) if len(measured)>1 else None,
                          first_discarded=rows[0][key] if len(rows)>1 else None,
                          values=[row[key] for row in measured])
                for key in ['f1_weighted','f1_macro','exact_match','request_errors',
                            'parse_failures','oov_predictions_after_parse']}
        (args.output/'correctness-aggregate.json').write_text(json.dumps(correctness_aggregate, indent=2)+'\n')
        phase = 'final-idle'
        stop_sampler.wait(2)
        progress('completed', performance_requests=400,
                 correctness_requests=args.correctness_rounds*2*expected_items,
                 model_quality_outliers_require_review=True)
    except Exception as exc:
        progress('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        stop_sampler.set()
        if 'sampler' in locals():
            sampler.join(timeout=30)
        if 'log' in locals():
            log.close()
        (args.output/'memory-samples.json').write_text(json.dumps(memory, indent=2)+'\n')
        memory_summary = {}
        for name in ['baseline', 'loading', 'ready', 'performance', 'post-performance-idle', 'correctness', 'final-idle']:
            rows = [row for row in memory if row['phase'] == name and 'GPU_used_MiB' in row]
            if rows:
                memory_summary[name] = dict(samples=len(rows), GPU_peak_MiB=max(row['GPU_used_MiB'] for row in rows),
                                            GPU_min_MiB=min(row['GPU_used_MiB'] for row in rows),
                                            host_RSS_peak_KiB=max(row['server_tree_host_RSS_KiB'] for row in rows))
        (args.output/'memory-summary.json').write_text(json.dumps(memory_summary, indent=2)+'\n')
        lock.close()


if __name__ == '__main__':
    main()
