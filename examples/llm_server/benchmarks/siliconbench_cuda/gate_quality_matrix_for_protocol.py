"""Finish an active correctness cell while holding only its CPU queue parent.

Wait for an explicit protocol decision. Preserve all completed outputs and the
active server. Restore the original queue if the bounded gate expires.
"""
import argparse,hashlib,json,os,signal,time
from pathlib import Path

def identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ['Z','X'] else (pid,int(fields[19]))
    except FileNotFoundError:return None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--command-sha256',required=True)
    p.add_argument('--active-case',required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-hold-seconds',type=int,default=3600)
    a=p.parse_args();assert 60<=a.max_hold_seconds<=7200 and not a.output.exists()
    case=a.matrix/a.active_case;assert case.resolve().is_relative_to(a.matrix.resolve())
    initial=json.loads((a.matrix/'progress.json').read_text())
    assert initial['controller_pid']==a.controller_pid and initial['current_case']==a.active_case
    settings=json.loads((case/'settings.json').read_text());assert settings['mode']=='correctness'
    assert settings['model']=='muse-glimmer-30b' and settings['correctness_rounds']==1
    raw=Path('/proc',str(a.controller_pid),'cmdline').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==a.command_sha256 and b'/run_matrix_24h.py\x00' in raw
    parent=identity(a.controller_pid);assert parent
    a.output.mkdir(parents=True);held=False;superseded=False
    def state(status,**fields):
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),queue_parent_identity=parent,
            active_case=str(case),active_inference_preserved=True,**fields),indent=2)+'\n');pending.replace(a.output/'progress.json')
    def interrupted(signum,frame):raise InterruptedError('Restore owned CPU queue on gate interruption')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        assert identity(a.controller_pid)==parent
        os.kill(a.controller_pid,signal.SIGSTOP);held=True
        end=time.monotonic()+5
        while Path('/proc',str(a.controller_pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='T':
            assert time.monotonic()<end;time.sleep(.02)
        fresh=json.loads((a.matrix/'progress.json').read_text())
        assert fresh['current_case']==a.active_case and identity(a.controller_pid)==parent
        (a.output/'original-matrix-progress.json').write_text(json.dumps(fresh,indent=2)+'\n')
        deadline=time.monotonic()+a.max_hold_seconds
        while True:
            current=json.loads((case/'progress.json').read_text())
            assert current['status'] not in ['failed','cancelled','aborted']
            complete=current['status']=='completed' and identity(current['controller_pid']) is None
            state('waiting-protocol-decision' if complete else 'waiting-active-correctness-cell',active_cell_complete=complete)
            if complete and a.decision.exists():break
            if time.monotonic()>=deadline:
                state('hold-expired-original-queue-resumed');return
            time.sleep(5)
        decision=json.loads(a.decision.read_text())
        assert decision['queue_parent_identity']==list(parent) and decision['active_case']==a.active_case
        assert decision['action'] in ['resume-original','supersede-original-quality-protocol']
        if decision['action']=='supersede-original-quality-protocol':
            assert decision['replacement_quality_root'] and decision['original_results_preserved']
            assert identity(a.controller_pid)==parent
            os.kill(a.controller_pid,signal.SIGTERM);os.kill(a.controller_pid,signal.SIGCONT);held=False
            end=time.monotonic()+60
            while identity(a.controller_pid)==parent:
                assert time.monotonic()<end;time.sleep(.1)
            superseded=True
        state('completed',decision=decision,original_queue_superseded=superseded,completed_original_cell_preserved=True)
    except Exception as e:
        state('failed',detail=str(e)[-1000:],original_queue_superseded=superseded);raise
    finally:
        if held and identity(a.controller_pid)==parent:os.kill(a.controller_pid,signal.SIGCONT)

if __name__=='__main__':main()
