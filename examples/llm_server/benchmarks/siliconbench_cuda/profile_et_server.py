"""Profile an unchanged ET server with selected original agent requests.

Diagnostic only: the corpus is a disclosed subset, and Nsight changes timing.
Four rounds discard the first. The main performance/quality records are not
read, modified or promoted by this tool. The GPU lock covers process teardown.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['wrapper', 'client', 'prompts', 'lock', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['wrapper', 'client', 'prompts']:
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--concurrency', type=int, choices=[1,16], default=1)
    p.add_argument('--python', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--server-env', action='append', default=[])
    p.add_argument('--memory-only', action='store_true',
                   help='Retain CUDA allocation and graph-launch events without expanding every graph node')
    p.add_argument('--deadline-utc', required=True)
    a = p.parse_args()
    cutoff = datetime.datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
    assert len(a.indices) == len(set(a.indices)) and all(i >= 0 for i in a.indices)
    assert not a.output.exists(), 'Preserve all earlier profiles'
    for name in ['wrapper', 'client', 'prompts']:
        assert hashlib.sha256(getattr(a, name).read_bytes()).hexdigest() == getattr(a, name + '_sha256')
    corpus = json.loads(a.prompts.read_text())
    assert isinstance(corpus, list) and len(corpus) == 100
    chosen = [corpus[i] for i in a.indices]
    assert len(chosen) >= a.concurrency, 'The subset must fill the requested client concurrency'
    assert all(set(['name', 'messages', 'max_tokens']) <= set(row) for row in chosen)
    a.output.mkdir(parents=True)
    subset = a.output / 'selected-original-prompts.json'
    subset.write_text(json.dumps(chosen) + '\n')
    env = dict(os.environ, SB_PORT=str(a.port), SB_PREFIX_CACHE='off',
               APPLEBENCH_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    for value in a.server_env:
        name, content = value.split('=', 1)
        assert name in ['SB_MODEL', 'SB_ET_CACHEOFF_ARTIFACT', 'SB_ET_PREFIX_ENTRIES', 'SB_CHECKPOINT']
        env[name] = content
    report = dict(scope=__doc__, selected_indices=a.indices,
                  selected_prompt_names=[row['name'] for row in chosen],
                  source_prompts_sha256=a.prompts_sha256, client_sha256=a.client_sha256,
                  wrapper_sha256=a.wrapper_sha256, rounds=4, discarded=[1], averaged=[2,3,4],
                  concurrency=a.concurrency, prefix_cache=False, native_server_unchanged=True,
                  profiling_not_primary_performance=True, measurements=[])
    report['memory_only'] = a.memory_only
    server = None
    stop = threading.Event()
    sampler = None
    memory = []
    phase = 'waiting'
    start = time.monotonic()

    def state(status, **fields):
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields)) + '\n')
        pending.replace(a.output / 'progress.json')

    def remaining():
        seconds = (cutoff - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        assert seconds > 0, 'User tuning cutoff reached'
        return min(seconds, 1200)

    def apps():
        return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()

    def sample():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                                 '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                memory.append(dict(elapsed_s=time.monotonic()-start, phase=phase, GPU_used_MiB=float(value)))
            except (ValueError, subprocess.SubprocessError):
                pass
            stop.wait(1)

    lock = a.lock.open('a')
    try:
        state('waiting-GPU-lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        remaining()
        assert not apps(), 'Existing GPU work must be left untouched'
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        command = ['nsys', 'profile', '--trace=cuda' if a.memory_only else '--trace=cuda,cublas,nvtx',
                   '--cuda-graph-trace=graph' if a.memory_only else '--cuda-graph-trace=node',
                   '--cuda-memory-usage=true', '--cuda-flush-interval=1000', '--sample=none',
                   '--cpuctxsw=none', '-o', str(a.output/'trace'), 'bash', str(a.wrapper), 'executorch']
        report['profile_command'] = command
        phase = 'loading'
        with (a.output / 'profile-server.log').open('x') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state('starting-profiled-server', server_pid=server.pid)
            deadline = time.monotonic() + remaining()
            while True:
                assert server.poll() is None, 'Profiled server exited; inspect its retained log'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/models', timeout=3) as response:
                        if json.load(response).get('data'):
                            break
                except Exception:
                    pass
                assert time.monotonic() < deadline, 'Profiled server did not become ready'
                time.sleep(2)
            for repetition in [1,2,3,4]:
                phase = 'round-' + str(repetition)
                state('profiling-agent-requests', server_pid=server.pid, round=repetition)
                target = a.output / ('repeat-' + str(repetition))
                target.mkdir()
                client = [a.python, '-B', str(a.client), '--port', str(a.port), '--framework', 'executorch',
                          '--split', 'agent', '--concurrency', str(a.concurrency), '--requests', str(len(chosen)),
                          '--warmup', '3', '--prompts', str(subset), '--output', str(target/'metrics.json'),
                          '--outputs', str(target/'outputs.jsonl')]
                with (target / 'client.log').open('x') as output:
                    subprocess.run(client, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=remaining())
                metrics = json.loads((target / 'metrics.json').read_text())['concurrency_results'][0]
                assert metrics['successful'] + metrics['failed'] == len(chosen)
                report['measurements'].append(metrics)
    except BaseException as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise
    finally:
        phase = 'teardown'
        if len(report['measurements']) == 4:
            state('finishing-profile-data', requests_completed=sum(
                row['successful'] for row in report['measurements']))
        if server is not None and server.poll() is None:
            assert os.getpgid(server.pid) == server.pid
            os.killpg(server.pid, signal.SIGINT)
            try:
                # Nsight may need several minutes to finalize allocation data.
                # Keep the GPU lock until the profiler and its children exit.
                server.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
        stop.set()
        if sampler:
            sampler.join(timeout=20)
        (a.output / 'memory-samples.json').write_text(json.dumps(memory, indent=2) + '\n')
        (a.output / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
        # This controller has held the GPU lock throughout the server lifetime.
        deadline = time.monotonic() + 60
        while apps():
            assert time.monotonic() < deadline, 'CUDA process remains; do not start another experiment'
            time.sleep(1)
    trace = a.output / 'trace.nsys-rep'
    assert trace.exists(), 'Nsight did not retain a profile; inspect the log'
    database = a.output / 'trace.sqlite'
    if not database.exists():
        with (a.output / 'nsys-export.log').open('x') as log:
            subprocess.run(['nsys', 'export', '--type=sqlite', '-o', str(database), str(trace)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    connection = sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True)
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    summary = dict(tables=tables, schemas={}, API_totals={}, native_server_unchanged=True,
                   trace_kept_remote=True, not_primary_performance=True)
    for table in tables:
        if table.startswith('CUPTI_ACTIVITY_KIND_'):
            columns = [row[1] for row in connection.execute('PRAGMA table_info("' + table + '")')]
            summary['schemas'][table] = columns
            if table in ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'] and all(x in columns for x in ['nameId', 'start', 'end']):
                rows = connection.execute('SELECT s.value, count(*), sum(a.end-a.start)/1e6 FROM "' + table + '" a JOIN StringIds s ON a.nameId=s.id GROUP BY a.nameId').fetchall()
                summary['API_totals'][table] = [dict(name=name, calls=count, elapsed_ms=elapsed) for name,count,elapsed in rows]
    graph = [row for values in summary['API_totals'].values() for row in values if 'GraphLaunch' in row['name']]
    summary['observed_graph_launch_calls'] = sum(row['calls'] for row in graph)
    summary['memory_related_tables'] = [t for t in tables if 'MEMORY' in t or 'ALLOCATION' in t]
    connection.close()
    (a.output / 'trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state('completed', observed_graph_launch_calls=summary['observed_graph_launch_calls'],
          request_errors=sum(row['failed'] for row in report['measurements']),
          not_primary_performance=True, native_server_unchanged=True)
    lock.close()


if __name__ == '__main__':
    main()
