"""Finish one active quality cell, then run a disjoint task-owned continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def live(pid):
    p = Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0] not in ['Z','X']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--controller-command-sha256',required=True)
    p.add_argument('--active-case',required=True)
    p.add_argument('--own-case-plan',type=Path,required=True)
    p.add_argument('--peer-case-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert command and Path(command[0]).is_absolute()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    old = json.loads((a.matrix/'progress.json').read_text())
    original_plan = json.loads((a.matrix/'matrix.json').read_text())
    current = a.matrix/a.active_case
    own = json.loads(a.own_case_plan.read_text())
    peer = json.loads(a.peer_case_plan.read_text())
    assert old['controller_pid'] == a.controller_pid and old['current_case'] == a.active_case
    assert a.active_case == 'correctness-agent-llamacpp-off-prefix-off-c1'
    assert own['model'] == peer['model'] == original_plan['model'] == 'muse-glimmer-30b'
    qset = {tuple(row) for row in original_plan['cases'] if row[0] == 'correctness'}
    retained = {('correctness','off',1,'llamacpp-off')}
    dset, eset = {tuple(x) for x in own['cases']}, {tuple(x) for x in peer['cases']}
    assert not (dset & eset or dset & retained or eset & retained)
    assert dset | eset | retained == qset and len(dset) == 9 and len(eset) == 10
    raw = (Path('/proc')/str(a.controller_pid)/'cmdline').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == a.controller_command_sha256
    assert str(a.matrix).encode() in raw and b'/resume_matrix_24h.py\x00' in raw
    settings = json.loads((current/'settings.json').read_text())
    assert settings['mode'] == 'correctness' and settings['concurrency'] == 1
    assert settings['correctness_rounds'] == 1 and settings['correctness_items'] == 1146
    def state(status, **fields):
        pending = a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,active_cell=str(current),
            previous_controller=a.controller_pid,**fields),indent=2)+'\n')
        pending.replace(a.output/'progress.json')
    held = False
    stopped = False
    try:
        # Hold only the CPU queue parent. Its active case, server and client keep running.
        os.kill(a.controller_pid,signal.SIGSTOP)
        held = True
        state('waiting-active-full-quality-cell',active_inference_not_stopped=True,
              retained_quality_cells=1,own_remaining_cells=9,peer_remaining_cells=10)
        until = time.monotonic()+14400
        while True:
            pr = json.loads((current/'progress.json').read_text())
            if pr['status'] == 'completed' and not live(pr['controller_pid']):
                break
            assert pr['status'] != 'failed', 'Preserve failure and restore original queue'
            assert time.monotonic() < until, 'Bounded cell handoff expired'
            time.sleep(10)
        for shot in [0,5]:
            target = current/f'correctness/round-1/{shot}shot'
            rows = [json.loads(x) for x in (target/'responses.jsonl').read_text().splitlines()]
            score = json.loads((target/'scores.json').read_text())
            assert len(rows) == len({str(x['id']) for x in rows}) == 1146
            assert score['n_scored'] == 1146 and score['n_missing_responses'] == 0
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        (a.output/'original-controller-progress.json').write_text(json.dumps(old,indent=2)+'\n')
        os.kill(a.controller_pid,signal.SIGTERM)
        os.kill(a.controller_pid,signal.SIGCONT)
        held = False
        until = time.monotonic()+90
        while live(a.controller_pid):
            assert time.monotonic() < until
            time.sleep(1)
        stopped = True
        state('running-disjoint-continuation',original_complete_cell_preserved=True,
              original_CPU_parent_terminal=True,peer_plan=str(a.peer_case_plan))
        subprocess.run(command,check=True)
        state('completed',original_complete_cell_preserved=True,disjoint_quality_cells=9,
              peer_quality_cells=10,existing_performance_not_repeated=True)
    except Exception as e:
        if held and live(a.controller_pid):
            os.kill(a.controller_pid,signal.SIGCONT)
        state('failed',error_type=type(e).__name__,detail=str(e)[-1200:],
              original_controller_stopped=stopped,held_parent_restored=held)
        raise


if __name__ == '__main__':
    main()
