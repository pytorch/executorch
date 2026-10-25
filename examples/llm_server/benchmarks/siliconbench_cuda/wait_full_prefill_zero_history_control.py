"""After existing full-server tests, run a zero-history numerical attribution.

Reuse both full artifacts and the existing probe. Do not export a new model,
restart a workflow, overlap server timing, or change an auxiliary failure into
passing. The control measures logits only, not performance.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workflow-job','CPU-build-progress','et-source','accepted-main-review','job-dir','lock']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    assert not a.job_dir.exists()
    a.job_dir.mkdir(parents=True)
    def state(status,**fields):
        pending=a.job_dir/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            numerical_only=True,model_promoted=False,new_KA=False,**fields),indent=2)+'\n')
        pending.replace(a.job_dir/'progress.json')
    try:
        state('waiting-existing-full-server-workflow')
        deadline=time.monotonic()+86400
        while True:
            s=json.loads((a.workflow_job/'progress.json').read_text())
            if s['status'] in ['completed-pending-independent-full-model-selection','failed']:
                break
            os.kill(s['controller_pid'],0)
            assert time.monotonic()<deadline,'Observation expired; never restart the full workflow'
            time.sleep(30)
        assert (a.workflow_job/'model-pair-exports.json').exists()
        (a.job_dir/'parent-terminal-state.json').write_text(json.dumps(s,indent=2)+'\n')
        command=[sys.executable,'-B',str(Path(__file__).with_name('run_public_prefill_mini_runtime.py')),
            '--workflow-job',str(a.workflow_job),'--workflow-pair-filename','model-pair-exports.json',
            '--expected-n-layers','52','--runtime-sessions','16','--history-tokens','0',
            '--CPU-build-progress',str(a.CPU_build_progress),'--et-source',str(a.et_source),
            '--accepted-main-review',str(a.accepted_main_review),'--job-dir',str(a.job_dir/'executor'),
            '--lock',str(a.lock),'--numerical-only','--continue-after-numerical-screen-failure']
        with (a.job_dir/'executor.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state('running-zero-history-control',child_pid=child.pid)
            try:
                assert child.wait(timeout=21600)==0
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=120)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        result=json.loads((a.job_dir/'executor/results.json').read_text())
        assert result['history_tokens']==0 and result['numerical_only'] and result['n_layers']==52
        state('completed-pending-independent-numerical-attribution',
              auxiliary_numeric_pass=result['mini_numeric_pass'])
    except BaseException as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1800:])
        raise


if __name__=='__main__':main()
