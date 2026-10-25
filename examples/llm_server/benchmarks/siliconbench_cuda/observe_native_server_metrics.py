"""Read stock vLLM stage metrics without modifying a benchmark or its server.

The observation covers the server lifetime, including all warmup requests.
Request stage durations include scheduling; they are not isolated GPU timers.
Incomplete windows remain labelled incomplete. No inference request is added.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request


METRICS = {
    'vllm:request_success_total',
    'vllm:request_prefill_time_seconds_count',
    'vllm:request_prefill_time_seconds_sum',
    'vllm:request_decode_time_seconds_count',
    'vllm:request_decode_time_seconds_sum',
    'vllm:request_prefill_kv_computed_tokens_count',
    'vllm:request_prefill_kv_computed_tokens_sum',
    'vllm:request_prompt_tokens_count',
    'vllm:request_prompt_tokens_sum',
    'vllm:request_generation_tokens_count',
    'vllm:request_generation_tokens_sum',
    'vllm:prompt_tokens_total',
    'vllm:generation_tokens_total',
    'vllm:prefix_cache_hits_total',
    'vllm:prefix_cache_queries_total',
}


def parse_metrics(raw):
    values = {}
    registered = {m[1] for m in re.finditer(r'^# TYPE ([^\s]+) (?:counter|histogram)$', raw, re.MULTILINE)}
    for name in METRICS:
        base = re.sub(r'_(?:sum|count)$', '', name)
        if base in registered:
            values[name] = {'value': 0, 'series': [], 'registered_without_series': True}
    pattern = re.compile(r'^([^\s{]+)(\{.*\})?\s+([^\s]+)(?:\s+.*)?$')
    for line in raw.splitlines():
        match = pattern.fullmatch(line)
        if not match or match[1] not in METRICS:
            continue
        number = float(match[3])
        assert math.isfinite(number) and number >= 0
        row = values.setdefault(match[1], {'value': 0, 'series': []})
        row.pop('registered_without_series', None)
        row['value'] += number
        row['series'].append({'labels': match[2] or '', 'value': number})
    return values


def identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ['Z', 'X'] else (pid, int(fields[19]))
    except FileNotFoundError:
        return None


def snapshot(pid):
    before = identity(pid)
    if before is None:
        raise ProcessLookupError(pid)
    argv = [x.decode() for x in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if x]
    assert '--port' in argv
    port = int(argv[argv.index('--port') + 1])
    raw = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=2).read().decode()
    assert identity(pid) == before
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'server_identity': before, 'port': port, 'values': parse_metrics(raw),
            'raw': raw, 'raw_sha256': hashlib.sha256(raw.encode()).hexdigest()}


def delta(start, end):
    assert start['server_identity'] == end['server_identity']
    out = {}
    for name in METRICS & start['values'].keys() & end['values'].keys():
        value = end['values'][name]['value'] - start['values'][name]['value']
        assert value >= 0, 'Counters reset inside the observed server identity'
        out[name] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrices', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=43200)
    args = parser.parse_args()
    assert args.max_seconds > 0 and not args.output.exists()
    args.output.mkdir(parents=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    cases = {}
    terminal = {}

    def save(status):
        pending = args.output / 'progress.pending.json'
        record = {'status': status, 'controller_pid': os.getpid(),
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'completed': sum('end' in c for c in cases.values()),
                  'source_sha256': source_sha, 'matrices': terminal,
                  'cases': {k: {x: v for x, v in c.items() if x not in ['start', 'end']}
                            for k, c in cases.items()},
                  'model_requests_added': 0, 'runtime_modified': False,
                  'scope': 'Server lifecycle including warmups; not primary rounds2-4 or pure GPU time'}
        pending.write_text(json.dumps(record, indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    save('observing-stock-server-metrics')
    while time.monotonic() - started < args.max_seconds:
        for matrix in args.matrices:
            progress_path = matrix / 'progress.json'
            if not progress_path.exists():
                terminal[str(matrix)] = 'waiting-for-existing-matrix'
                continue
            state = json.loads(progress_path.read_text())
            terminal[str(matrix)] = state['status']
            for done in state.get('completed', []):
                key = str(matrix / done)
                if key in cases and 'end' not in cases[key]:
                    cases[key]['terminal_without_final_snapshot'] = True
            name = state.get('current_case')
            if not name or not name.startswith('performance-agent-vllm-'):
                continue
            case = matrix / name
            path = case / 'progress.json'
            if not path.exists():
                continue
            p = json.loads(path.read_text())
            pid = p.get('server_pid')
            if not pid or identity(pid) is None:
                continue
            key = str(case)
            record = cases.setdefault(key, {'case': key, 'snapshots': 0})
            try:
                if 'start' not in record:
                    record['start'] = snapshot(pid)
                    record['snapshots'] += 1
                    initial = record['start']['values'].get('vllm:request_success_total', {}).get('value')
                    record['baseline_completed_requests'] = initial
                    (args.output / (hashlib.sha256(key.encode()).hexdigest()[:16] + '-start.json')).write_text(
                        json.dumps(record['start'], indent=2) + '\n')
                if 'end' not in record and (case / 'performance-aggregate.json').exists():
                    final = snapshot(pid)
                    record['snapshots'] += 1
                    record['end'] = final
                    record['delta'] = delta(record['start'], final)
                    count = record['delta'].get('vllm:request_success_total')
                    phase_counts = [record['delta'].get(n) for n in [
                        'vllm:request_prefill_time_seconds_count', 'vllm:request_decode_time_seconds_count']]
                    record['whole_400_plus12_warmup_window_verified'] = (
                        record['baseline_completed_requests'] == 0 and count == 412 and phase_counts == [412, 412])
                    record['partial_window'] = not record['whole_400_plus12_warmup_window_verified']
                    record['whole_cumulative_token_counter_window_verified'] = (
                        record['whole_400_plus12_warmup_window_verified'] and all(
                            record['start']['values'].get(n, {}).get('value') == 0
                            for n in ['vllm:prompt_tokens_total', 'vllm:generation_tokens_total']))
                    record['primary_mean_rounds2_4_timer'] = False
                    record['pure_GPU_timer'] = False
                    filename = hashlib.sha256(key.encode()).hexdigest()[:16] + '-end.json'
                    (args.output / filename).write_text(json.dumps(record, indent=2) + '\n')
            except (OSError, ValueError, AssertionError) as exc:
                record['last_observation_error'] = {'type': type(exc).__name__, 'detail': str(exc)}
        save('observing-stock-server-metrics')
        if terminal and all(terminal.get(str(m)) in ['completed', 'failed'] for m in args.matrices):
            save('completed')
            return
        time.sleep(.5)
    save('observation-time-limit')


if __name__ == '__main__':
    main()
