"""Select completed ET prefix policy probes and collect the main server matrix.

Selection is based on error-free c1/c16 performance, with provenance preserved.
Quality acceptance is pending until full and sampled F1 results are reviewed.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


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
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous selection and collection attempts'
    args.output.mkdir(parents=True)
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
                    subprocess.run(['supervisorctl', 'stop', name], capture_output=True, text=True, timeout=90, check=True)
                state('tuning-cutoff-reached', stopped_pending_programs=[p['program'] for p in pending])
                break
            time.sleep(30)
        candidates = {'off': [], 'on': []}
        rejected = []
        for probe in probes:
            output = Path(probe['output'])
            progress = output/'progress.json'
            if not progress.exists() or json.loads(progress.read_text())['status'] != 'completed':
                rejected.append(dict(program=probe['program'], reason='Probe did not complete; retained as an unsuccessful attempt'))
                continue
            provenance = json.loads((output/'evaluation-provenance.json').read_text())
            assert provenance['maximum_context'] == 131072 and provenance['initial_kv_capacity'] == 512
            wrapper = next(p for p in output.iterdir() if p.name in ['serve-prefix-wrapper.sh', 'serve-policy.sh'])
            matrix = output/'matrix' if (output/'matrix').exists() else output
            cells = []
            for cell in matrix.iterdir():
                if not cell.is_dir() or not (cell/'performance-aggregate.json').exists():
                    continue
                settings = json.loads((cell/'settings.json').read_text())
                aggregate = json.loads((cell/'performance-aggregate.json').read_text())
                memory = json.loads((cell/'memory-summary.json').read_text())
                assert settings['mode']=='performance' and settings['performance_rounds']==4
                assert settings['concurrency'] in [1,16]
                if any(v != 100 for v in aggregate['successful']['values']) or any(aggregate['failed']['values']):
                    rejected.append(dict(program=probe['program'], cell=cell.name, reason='Request failures'))
                    continue
                cells.append(dict(cache='on' if settings['prefix_cache'] else 'off', concurrency=settings['concurrency'],
                                  TPS=aggregate['output_throughput_tps']['mean'], cell=str(cell),
                                  peak_MiB=max(v['GPU_peak_MiB'] for v in memory.values())))
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
