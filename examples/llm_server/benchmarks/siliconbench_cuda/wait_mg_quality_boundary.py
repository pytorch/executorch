"""Finish the stock performance queue, then supersede only its quality queue.

The first active quality cell is preserved. The existing tested CPU gate holds
the matrix parent and waits for that cell to finish before any replacement.
This does not change a server, performance cell, model, label or scorer.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from gate_quality_matrix_for_protocol import identity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--command-sha256',required=True)
    p.add_argument('--replacement-quality-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    parent=identity(a.controller_pid);assert parent
    def verify():
        assert identity(a.controller_pid)==parent,'Original queue must remain the same live process'
        raw=Path('/proc',str(a.controller_pid),'cmdline').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==a.command_sha256 and b'/run_matrix_24h.py\x00' in raw
        state=json.loads((a.matrix/'progress.json').read_text())
        assert state['controller_pid']==a.controller_pid and state['status']=='running'
        return state
    verify();a.output.mkdir(parents=True)
    def state(status,**fields):
        path=a.output/'progress.pending.json'
        path.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),queue_parent_identity=parent,
            original_performance_preserved=True,**fields),indent=2)+'\n');path.replace(a.output/'progress.json')
    try:
        while True:
            current=verify();case=current['current_case']
            state('waiting-original-performance-completion',current_case=case,completed=current['completed'])
            if case.startswith('correctness-agent-'):break
            assert case.startswith('performance-agent-')
            time.sleep(20)
        plan=json.loads((a.matrix/'matrix.json').read_text());assert plan['model']=='muse-glimmer-30b'
        perf=[f'{phase}-agent-{variant}-prefix-{cache}-c{c}' for phase,cache,c,variant in plan['cases'] if phase=='performance']
        assert len(perf)==20 and set(perf)<=set(current['completed'])
        for name in perf:assert json.loads((a.matrix/name/'progress.json').read_text())['status']=='completed'
        decision=a.output/'decision.json'
        decision.write_text(json.dumps(dict(queue_parent_identity=parent,active_case=case,
            action='supersede-original-quality-protocol',replacement_quality_root=str(a.replacement_quality_root),
            original_results_preserved=True),indent=2)+'\n')
        command=[sys.executable,'-B',str(Path(__file__).with_name('gate_quality_matrix_for_protocol.py')),
            '--matrix',str(a.matrix),'--controller-pid',str(a.controller_pid),'--command-sha256',a.command_sha256,
            '--active-case',case,'--decision',str(decision),'--output',str(a.output/'gate'),'--max-hold-seconds','7200']
        state('waiting-active-original-quality-cell',current_case=case,all20_performance_cells_preserved=True)
        subprocess.run(command,check=True)
        result=json.loads((a.output/'gate/progress.json').read_text())
        assert result['status']=='completed' and result['original_queue_superseded']
        state('completed',original_queue_superseded=True,all20_performance_cells_preserved=True,
            first_original_quality_cell_preserved=True,replacement_quality_root=str(a.replacement_quality_root))
    except Exception as e:
        state('failed',detail=str(e)[-1200:]);raise

if __name__=='__main__':main()
