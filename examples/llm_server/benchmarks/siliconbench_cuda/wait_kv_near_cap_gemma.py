"""Queue isolated near-cap KV probes and a Gemma1024 server diagnostic.

All work waits for named existing jobs, then uses the established GPU lock.
The full-size probe contains placeholder residency, not model weights. Only
after prefix-preservation checks does the actual existing1024 model run.
Neither probes nor diagnostic performance automatically select an ET model.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(16*1024*1024):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['main-job','after-job','et-source','build-inputs','checkpoint-build',
              'base-wrapper','artifact','client','prompts','correctness-root','lock','job-dir']:
        p.add_argument('--'+n,type=Path,required=True)
    for n in ['base-wrapper','PTE','PTD','client','prompts']:
        p.add_argument('--'+n+'-sha256',required=True)
    p.add_argument('--recorded-et-source',required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args();assert not a.job_dir.exists();a.job_dir.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**kw):
        tmp=a.job_dir/'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            new_KA=False,model_promoted=False,primary_results_changed=False,**kw),indent=2)+'\n')
        tmp.replace(a.job_dir/'progress.json')
    def run(cmd,log_path,phase,timeout,env=None):
        with log_path.open('x') as log:
            log.write(json.dumps(cmd)+'\n');log.flush()
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            state(phase,child_pid=child.pid,command=cmd)
            try:
                assert child.wait(timeout=timeout)==0, 'Own child failed; retain its log'
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=420)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
    try:
        state('waiting-named-full-workflow-and-runtime-profile')
        terminal={'completed-pending-independent-full-model-selection',
                  'completed-pending-independent-runtime-coverage-review','failed'}
        deadline=time.monotonic()+86400
        while True:
            parents={str(d):json.loads((d/'progress.json').read_text())
                     for d in [a.main_job,a.after_job]}
            waiting=[s for s in parents.values() if s['status'] not in terminal]
            if not waiting:break
            for s in waiting:os.kill(s['controller_pid'],0)
            assert time.monotonic()<deadline,'Wait expired; never restart another job'
            time.sleep(30)
        (a.job_dir/'parent-terminal-states.json').write_text(json.dumps(parents,indent=2)+'\n')
        for path,expected in [(a.base_wrapper,a.base_wrapper_sha256),
             (a.artifact/'model.pte',a.PTE_sha256),(a.artifact/'aoti_cuda_blob.ptd',a.PTD_sha256),
             (a.client,a.client_sha256),(a.prompts,a.prompts_sha256)]:
            assert digest(path)==expected,str(path)
        probe_results=[]
        with a.lock.open('a') as lock:
            state('waiting-established-GPU-lock');fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader'],text=True).strip(),'Leave other GPU work untouched'
            build=a.job_dir/'CPU-build'
            cmd=[sys.executable,'-B',str(package/'prepare_kv_near_cap_build.py'),
                 '--et-source',str(a.et_source),'--build-inputs',str(a.build_inputs),
                 '--checkpoint-build',str(a.checkpoint_build),'--output',str(build),
                 '--recorded-et-source',a.recorded_et_source]
            run(cmd,a.job_dir/'CPU-build-controller.log','running-isolated-CPU-build',1800,
                dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
            audit=json.loads((build/'build-audit.json').read_text())
            assert audit['status']=='CPU-build-completed'
            for round_number,order in enumerate([['baseline','candidate'],['candidate','baseline'],
                                                  ['baseline','candidate'],['candidate','baseline']],1):
                for scope in ['mini','full']:
                    for arm in order:
                        binary=Path(audit['binaries'][arm]['probe'])
                        assert digest(binary)==audit['binaries'][arm]['probe_SHA256']
                        log=a.job_dir/f'probe-r{round_number}-{scope}-{arm}.log'
                        run([str(binary),scope,arm],log,'running-KV-growth-probe',180)
                        result=json.loads(log.read_text().splitlines()[-1]);result['round']=round_number
                        assert result['prefix_preserved'] and not result['model_loaded']
                        if scope=='mini':assert result['full_buffers_checked']
                        if arm=='candidate':assert all(s['error']==0 for s in result['stages'])
                        if scope=='full' and arm=='baseline' and result['baseline_expected_OOM_observed']:
                            assert result['stages'][-1]['rollback_pointers_unchanged']
                        probe_results.append(result)
                        (a.job_dir/'probe-results.json').write_text(json.dumps(probe_results,indent=2)+'\n')
            state('KV-probes-completed-model-validation-pending',probe_records=len(probe_results))
        # All probe children have exited before releasing the GPU lock.
        text=a.base_wrapper.read_text()
        artifact_re=re.compile(r'(?m)^([ \t]*)artifact=[^\n]+$')
        worker_re=re.compile(r'(--worker-bin\s+)("[^"]+"|\S+)')
        assert len(artifact_re.findall(text))==len(worker_re.findall(text))==1
        text=artifact_re.sub(lambda m:m[1]+'artifact='+shlex.quote(str(a.artifact)),text)
        text=worker_re.sub(lambda m:m[1]+shlex.quote(audit['binaries']['candidate']['worker']),text)
        wrapper=a.job_dir/'serve-near-cap1024.sh';wrapper.write_text(text);wrapper.chmod(0o700)
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        protocol=dict(wrapper=str(wrapper),wrapper_SHA256=digest(wrapper),
            original_wrapper=str(a.base_wrapper),original_wrapper_SHA256=a.base_wrapper_sha256,
            artifact=str(a.artifact),PTE_SHA256=a.PTE_sha256,PTD_SHA256=a.PTD_sha256,
            client=str(a.client),client_SHA256=a.client_sha256,prompts_SHA256=a.prompts_sha256,
            only_wrapper_changes=['artifact512-to-existing1024','isolated-near-cap-worker'],
            bulk_A16=True,static_graphs=True,active_sessions=16,decode_limit=16,
            logical_context=131072,initial_KV=512,model_promoted=False)
        (a.job_dir/'server-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        cmd=[a.python,'-B',str(package/'run_matrix_24h.py'),
             '--wrapper',str(wrapper),'--model','gemma4-12b-it','--client',str(a.client),
             '--prompts',str(a.prompts),'--correctness-root',str(a.correctness_root),
             '--python',a.python,'--output',str(a.job_dir/'server-diagnostic'),
             '--lock',str(a.lock),'--port',str(a.port),'--variants','executorch',
             '--cache-modes','off','--concurrencies','16','1','--phases','performance']
        run(cmd,a.job_dir/'server-diagnostic-controller.log','running-actual1024-server-diagnostic',14400)
        final=json.loads((a.job_dir/'server-diagnostic/progress.json').read_text())
        assert final['status']=='completed' and len(final['completed'])==2
        state('completed-pending-independent-probe-and-server-review',probe_records=len(probe_results),
              performance_cells=2,complete_model_F1_not_yet_tested=True)
    except BaseException as e:
        state('failed',error_type=type(e).__name__,detail=str(e)[-1800:]);raise


if __name__=='__main__':main()
