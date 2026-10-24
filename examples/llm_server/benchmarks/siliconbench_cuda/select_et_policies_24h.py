"""Select completed ET prefix policy probes and collect the main server matrix.

Selection is based on error-free c1/c16 performance, with provenance preserved.
Quality acceptance is pending until full and sampled F1 results are reviewed.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time


TERMINAL_SERVICES = {'EXITED', 'FATAL', 'STOPPED'}


def live(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0] not in ['Z', 'X']
    except FileNotFoundError:
        return False


def stop_at_cutoff(name):
    """Handle a concurrent deadline guard stopping this task-owned probe."""
    assert name.startswith('sb5090') and '24h-policy' in name
    before = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True, timeout=30)
    if TERMINAL_SERVICES.intersection(before.stdout.split()):
        return dict(program=name, action='already-terminal', status=before.stdout.strip())
    assert {'RUNNING', 'STARTING', 'STOPPING'}.intersection(before.stdout.split()), before.stdout
    stopped = subprocess.run(['supervisorctl', 'stop', name], capture_output=True, text=True, timeout=90)
    after = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True, timeout=30)
    assert TERMINAL_SERVICES.intersection(after.stdout.split()), (name, stopped.stdout, after.stdout)
    return dict(program=name, action='terminal-after-cutoff-stop', stop_returncode=stopped.returncode,
                status=after.stdout.strip(), concurrent_guard_stop_allowed=True)


def audit_cell(cell, model, wrapper, client, prompts):
    """Audit every round and HTTP usage before using a completed policy cell."""
    progress = json.loads((cell/'progress.json').read_text())
    assert progress['status']=='completed' and not live(progress['controller_pid'])
    settings = json.loads((cell/'settings.json').read_text())
    assert settings['mode']=='performance' and settings['model']==model
    assert settings['performance_rounds']==4 and settings['discarded']==[1] and settings['averaged']==[2,3,4]
    for label, path in [('wrapper', wrapper), ('client', client), ('prompts', prompts)]:
        assert settings[label+'_sha256']==hashlib.sha256(path.read_bytes()).hexdigest(), label
    samples = []
    for repetition in [1,2,3,4]:
        folder = cell/f'repeat-{repetition}'
        metrics = json.loads((folder/'metrics.json').read_text())['concurrency_results'][0]
        rows = [json.loads(line) for line in (folder/'outputs.jsonl').open()]
        assert metrics['successful']==100 and metrics['failed']==0 and metrics['server_usage_available']
        assert len(rows)==100 and {r['request_idx'] for r in rows}==set(range(100))
        for field, key in [('completion_tokens', 'total_output_tokens'), ('prompt_tokens', 'total_input_tokens')]:
            assert all(isinstance(r[field], int) and r[field]>=0 for r in rows)
            assert sum(r[field] for r in rows)==metrics[key]
        assert math.isclose(metrics['output_throughput_tps'], metrics['total_output_tokens']/metrics['wall_time_s'], rel_tol=1e-9)
        samples.append(metrics)
    aggregate = json.loads((cell/'performance-aggregate.json').read_text())
    for key, value in aggregate.items():
        measured = [r[key] for r in samples[1:]]
        assert value['first_discarded']==samples[0][key] and value['values']==measured
        assert math.isclose(value['mean'], statistics.mean(measured), rel_tol=1e-9, abs_tol=1e-10)
        assert math.isclose(value['sample_stddev'], statistics.stdev(measured), rel_tol=1e-9, abs_tol=1e-10)
    memory_path = cell/'performance-memory-aggregate.json'
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
    else:
        # Older probes retain original samples but lack explicit round tags.
        # Reconstruct the same disclosed one-second boundaries in memory only.
        anchor = (cell/'progress.json').stat().st_mtime-progress['elapsed_s']
        ends = [(cell/f'repeat-{n}/metrics.json').stat().st_mtime for n in [1,2,3,4]]
        assert ends==sorted(ends)
        recorded = json.loads((cell/'memory-samples.json').read_text())
        rounds = []
        for n in [1,2,3,4]:
            low = ends[n-2] if n>1 else float('-inf')
            observed = [r for r in recorded if r.get('phase')=='performance' and 'GPU_used_MiB' in r
                        and low<anchor+r['elapsed_s']<=ends[n-1]]
            assert observed, 'Missing recorded round memory observations'
            rounds.append(dict(round=n, samples=len(observed), GPU_peak_MiB=max(r['GPU_used_MiB'] for r in observed),
                               host_RSS_peak_KiB=max(r['server_tree_host_RSS_KiB'] for r in observed)))
        memory = {key:dict(first_discarded=rounds[0][key], values=[r[key] for r in rounds[1:]],
                          mean=statistics.mean(r[key] for r in rounds[1:]), sample_stddev=statistics.stdev(r[key] for r in rounds[1:]))
                  for key in ['GPU_peak_MiB', 'host_RSS_peak_KiB']}
        memory.update(rounds=rounds, provenance='Read-only reconstruction from original one-second sample and metric timestamps')
    assert [r['round'] for r in memory['rounds']]==[1,2,3,4]
    for key in ['GPU_peak_MiB', 'host_RSS_peak_KiB']:
        values = [r[key] for r in memory['rounds'][1:]]
        assert memory[key]['first_discarded']==memory['rounds'][0][key]
        assert memory[key]['values']==values
        assert math.isclose(memory[key]['mean'], statistics.mean(values), rel_tol=1e-9, abs_tol=1e-10)
        assert math.isclose(memory[key]['sample_stddev'], statistics.stdev(values), rel_tol=1e-9, abs_tol=1e-10)
    return settings, aggregate, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19088)
    parser.add_argument('--resume-waiting-controller-pid', type=int,
                        help='Resume only an existing waiting selector after its original process has exited')
    args = parser.parse_args()
    if args.resume_waiting_controller_pid is None:
        assert not args.output.exists(), 'Preserve previous selection and collection attempts'
        args.output.mkdir(parents=True)
    else:
        previous = json.loads((args.output/'progress.json').read_text())
        assert previous['status']=='waiting-performance-policy-probes'
        assert previous['controller_pid']==args.resume_waiting_controller_pid and not live(previous['controller_pid'])
        assert not any((args.output/name).exists() for name in ['selection.json', 'performance', 'correctness'])
        archive = args.output/('preserved-waiting-controller-'+str(previous['controller_pid'])+'.json')
        assert not archive.exists()
        archive.write_text(json.dumps(dict(previous_progress=previous, resumed_by=os.getpid(),
                            resume_only_before_collection=True), indent=2)+'\n')
    plan = json.loads(args.probe_plan.read_text())
    probes = plan['probes']
    deadline = datetime.datetime.fromisoformat(plan['tuning_deadline_utc'].replace('Z', '+00:00'))

    def state(status, **fields):
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields), indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    try:
        state('waiting-performance-policy-probes')
        while True:
            pending = []
            for probe in probes:
                path = Path(probe['output'])/'progress.json'
                progress = json.loads(path.read_text()) if path.exists() else None
                service = subprocess.run(['supervisorctl', 'status', probe['program']], capture_output=True, text=True, timeout=30).stdout.split()
                terminal = progress and progress['status'] in ['completed', 'failed', 'cancelled', 'aborted']
                terminal = terminal or any(value in service for value in ['EXITED', 'FATAL', 'STOPPED'])
                if not terminal:
                    pending.append(probe)
            if not pending:
                break
            if datetime.datetime.now(datetime.timezone.utc) >= deadline:
                for probe in pending:
                    name = probe['program']
                    assert name.startswith('sb5090') and '24h-policy' in name
                    stop_at_cutoff(name)
                state('tuning-cutoff-reached', stopped_pending_programs=[p['program'] for p in pending])
                break
            time.sleep(30)
        candidates = {'off': [], 'on': []}
        rejected = []
        for probe in probes:
            output = Path(probe['output'])
            progress = output/'progress.json'
            if not progress.exists():
                rejected.append(dict(program=probe['program'], reason='Probe produced no collection state'))
                continue
            probe_state = json.loads(progress.read_text())
            if probe_state['status'] != 'completed':
                rejected.append(dict(program=probe['program'], reason='Probe overall incomplete; only individually completed audited c1/c16 cache pairs are eligible',
                                     overall_status=probe_state['status']))
            provenance_path = output/'evaluation-provenance.json'
            wrappers = [p for p in output.iterdir() if p.name in ['serve-prefix-wrapper.sh', 'serve-policy.sh']]
            if not provenance_path.exists() or len(wrappers)!=1:
                rejected.append(dict(program=probe['program'], reason='Preparation/provenance incomplete; no policy cells eligible'))
                continue
            provenance = json.loads(provenance_path.read_text())
            if provenance.get('maximum_context')!=131072 or provenance.get('initial_kv_capacity')!=512:
                rejected.append(dict(program=probe['program'], reason='Context/initial KV provenance mismatch'))
                continue
            wrapper = wrappers[0]
            matrix = output/'matrix' if (output/'matrix').exists() else output
            cells = []
            for cell in matrix.iterdir():
                if not cell.is_dir() or not (cell/'performance-aggregate.json').exists():
                    continue
                try:
                    settings, aggregate, memory = audit_cell(cell, args.model, wrapper, args.client, args.prompts)
                except (AssertionError, FileNotFoundError, KeyError) as error:
                    rejected.append(dict(program=probe['program'], cell=cell.name, reason='Cell failed complete four-round audit', detail=str(error)[-400:]))
                    continue
                assert settings['mode']=='performance' and settings['performance_rounds']==4
                assert settings['concurrency'] in [1,16]
                if any(v != 100 for v in aggregate['successful']['values']) or any(aggregate['failed']['values']):
                    rejected.append(dict(program=probe['program'], cell=cell.name, reason='Request failures'))
                    continue
                cells.append(dict(cache='on' if settings['prefix_cache'] else 'off', concurrency=settings['concurrency'],
                                  TPS=aggregate['output_throughput_tps']['mean'], cell=str(cell),
                                  peak_MiB=memory['GPU_peak_MiB']['mean'], all400_requests_audited=True,
                                  four_round_memory_audited=True, overall_probe_status=probe_state['status']))
            for cache in ['off', 'on']:
                rows = [row for row in cells if row['cache']==cache]
                if len(rows)!=2 or {row['concurrency'] for row in rows}!={1,16} or any(row['TPS']<=0 for row in rows):
                    continue
                score = math.exp(sum(math.log(row['TPS']) for row in rows)/2)
                candidates[cache].append(dict(score=score, wrapper=str(wrapper), probe=probe, c1_c16=rows, provenance=provenance))
        assert all(candidates.values()), 'No completed error-free policy for one cache mode; inspect preserved export/probe failures'
        selected = {cache: max(rows, key=lambda row: row['score']) for cache, rows in candidates.items()}
        (args.output/'selection.json').write_text(json.dumps(dict(selected=selected, candidates=candidates, rejected=rejected,
                 selection_metric='Geometric mean of output TPS at c1 and c16, within each cache mode',
                 quality_acceptance_pending=True, kernel_acceptance_is_separate=True), indent=2)+'\n')
        state('collecting-main-matrix', selected_programs={k:v['probe']['program'] for k,v in selected.items()})
        for phase in ['performance', 'correctness']:
            for cache, chosen in selected.items():
                probe = chosen['probe']
                env = dict(os.environ, **probe.get('environment', {}))
                env['SB_ET_PREFIX_ENTRIES'] = str(probe['prefix_entries'])
                env['SB_CHECKPOINT'] = str(probe['checkpoint_interval'] if cache=='on' else 0)
                cmd = [args.python, '-B', str(Path(__file__).with_name('run_matrix_24h.py')),
                       '--wrapper', chosen['wrapper'], '--model', args.model, '--client', str(args.client),
                       '--prompts', str(args.prompts), '--correctness-root', str(args.correctness_root), '--python', args.python,
                       '--output', str(args.output/phase/cache), '--lock', str(args.lock), '--port', str(args.port),
                       '--phases', phase, '--variants', 'executorch', '--cache-modes', cache, '--concurrencies', '1', '2', '4', '8', '16']
                with (args.output/(phase+'-'+cache+'-matrix.log')).open('x') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert json.loads((args.output/phase/cache/'progress.json').read_text())['status']=='completed'
        state('completed', performance_cells=10, full_quality_cells=4, smoke_quality_cells=6,
              final_quality_memory_and_kernel_acceptance_pending=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
