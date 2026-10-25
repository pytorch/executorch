"""Queue a runtime kernel-coverage diagnostic after full server qualification.

Reuse the exported candidate and existing server/client without modifying
them. Nsight results describe runtime route coverage; they are never promoted
to primary wall-time results. The established GPU lock covers the diagnostic.
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
    for name in ['workflow-job','wrapper','client','prompts','lock','job-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['wrapper','client','prompts']:
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--indices',type=int,nargs='+',required=True)
    a=p.parse_args()
    assert not a.job_dir.exists()
    a.job_dir.mkdir(parents=True)

    def state(status,**fields):
        pending=a.job_dir/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            runtime_kernel_coverage_only=True,primary_results_unmodified=True,
            model_promoted=False,new_KA=False,**fields),indent=2)+'\n')
        pending.replace(a.job_dir/'progress.json')

    try:
        state('waiting-existing-full-server-workflow')
        deadline=time.monotonic()+86400
        while True:
            s=json.loads((a.workflow_job/'progress.json').read_text())
            if s['status']=='completed-pending-independent-full-model-selection':break
            assert s['status']!='failed',s
            os.kill(s['controller_pid'],0)
            assert time.monotonic()<deadline,'Observation expired; never restart a workflow'
            time.sleep(30)
        (a.job_dir/'parent-terminal-state.json').write_text(json.dumps(s,indent=2)+'\n')
        cutoff=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)
        command=[sys.executable,'-B',str(Path(__file__).with_name('profile_et_server.py')),
            '--wrapper',str(a.wrapper),'--wrapper-sha256',a.wrapper_sha256,
            '--client',str(a.client),'--client-sha256',a.client_sha256,
            '--prompts',str(a.prompts),'--prompts-sha256',a.prompts_sha256,
            '--lock',str(a.lock),'--output',str(a.job_dir/'profile'),
            '--indices',*[str(i) for i in a.indices],
            '--concurrency','16','--python',a.python,'--port',str(a.port),
            '--deadline-utc',cutoff.isoformat()]
        (a.job_dir/'profile-command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (a.job_dir/'profile-controller.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state('running-candidate-kernel-coverage',child_pid=child.pid)
            try:
                assert child.wait(timeout=7800)==0
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=420)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        result=json.loads((a.job_dir/'profile/progress.json').read_text())
        assert result['status']=='completed' and result['request_errors']==0
        state('completed-pending-independent-runtime-coverage-review',
              observed_graph_launch_calls=result['observed_graph_launch_calls'])
    except BaseException as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1800:])
        raise


if __name__=='__main__':main()
