"""Complete the bounded larger-M roofline experiment on one GPU.

Wait for the existing baseline comparison and fine Gate crossover screen.
Revalidate all three selected policies on this GPU, export a separate model,
measure baseline/candidate/stock default llama.cpp, then collect a profile.
No model is promoted and no original main result is changed.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['fine-gate-job','baseline-export-job','baseline-artifact','after-comparison',
              'ET-SDK-job','CPP-SDK-job','gguf','et-source','reference-deployment',
              'policies','lock','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--attention-source-sha256',required=True)
    p.add_argument('--deadline-UTC',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    deadline=datetime.datetime.fromisoformat(a.deadline_UTC)
    assert deadline.tzinfo is not None
    package=Path(__file__).resolve().parent
    def remaining():
        value=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        assert value>0,'Experiment deadline reached; preserve unfinished outputs'
        return value
    def state(status,**extra):
        tmp=a.output/'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_UTC=a.deadline_UTC,
            main_results_changed=False,model_promoted=False,**extra),indent=2)+'\n')
        tmp.replace(a.output/'progress.json')
    def wait(job,accepted):
        while True:
            remaining();f=job/'progress.json'
            if f.exists():
                v=json.loads(f.read_text())
                if v['status']==accepted:return v
                assert v['status']!='failed',v
                os.kill(v['controller_pid'],0)
            time.sleep(15)
    def run(cmd,label):
        with (a.output/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(v) for v in cmd],stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            state('running-'+label,child_pid=child.pid)
            try:assert child.wait(timeout=remaining())==0,label
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        state('waiting-existing-baseline-and-fine-screen')
        wait(a.baseline_export_job,'completed')
        wait(a.fine_gate_job,'completed-operator-screen-pending-full-forward-validation')
        wait(a.after_comparison,'completed-pending-independent-review')
        request=json.loads(a.policies.read_text())
        assert set(request['policies'])=={'ffn_down','ffn_gate_up','attention_out'}
        policies=copy.deepcopy(request['policies'])
        assert request['GGUF_SHA256']==sha(a.gguf)
        for role,policy in policies.items():
            assert policy['actual_row_grid'] and len(policy['packed_SHA256'])==5
        fine=json.loads((a.fine_gate_job/'results.json').read_text())
        assert fine['actual_row_grid']
        cases=[c for c in fine['cases'] if c['role']=='ffn_gate_up']
        assert cases and all(c['M']>=520 and c['packed_SHA256']==policies['ffn_gate_up']['packed_SHA256'] for c in cases)
        winners=[c for c in cases if c['candidate_over_complete_fallback']<1]
        assert winners,'No fine Gate winner; leave this candidate unexported'
        chosen=max(winners,key=lambda c:c['M'])
        assert chosen['M']<1024,'The measured 1024 Gate regression must retain fallback'
        old_limit=policies['ffn_gate_up']['branches'][-2]['max_M']
        assert chosen['M']>old_limit
        policies['ffn_gate_up']['branches'][-1]=dict(max_M=chosen['M'],bucket=chosen['bucket'],
            split_k=chosen['winner']['split_k'],config=chosen['winner']['config'])
        selection=dict(policies=policies,fine_screen_SHA256=sha(a.fine_gate_job/'results.json'),
            chosen_gate_M=chosen['M'],gate_native_over_fallback=chosen['candidate_over_complete_fallback'],
            source_policy_SHA256=sha(a.policies),main_results_changed=False)
        (a.output/'policy-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        reports=[]
        for role in ['ffn_down','ffn_gate_up','attention_out']:
            policy_file=a.output/(role+'-policy.json');policy_file.write_text(json.dumps(policies[role],indent=2)+'\n')
            job=a.output/(role+'-dynamic-AOTI')
            run([sys.executable,'-B',package/'benchmark_roofline_large_w4a16_aoti.py',
                 '--et-source',a.et_source,'--gguf',a.gguf,'--policy',policy_file,
                 '--output',job,'--lock',a.lock,'--role',role,'--max-M','1024'],role+'-dynamic-AOTI')
            report=job/'results.json';value=json.loads(report.read_text())
            assert json.loads((job/'progress.json').read_text())['status']=='completed-pending-full-forward'
            assert value['policy']==policies[role] and value['dynamic_range']==[65,1024]
            assert all(c['deterministic'] and c['native_or_fallback_oracle']['finite'] and
                c['native_or_fallback_oracle']['relative_RMS']<.005 for c in value['cases'])
            reports.append(str(report))
        combined=a.output/'qualified-policies.json'
        combined.write_text(json.dumps(dict(policies=[policies[r] for r in ['ffn_down','ffn_gate_up','attention_out']],
            validation_reports=reports),indent=2)+'\n')
        artifact=a.output/'candidate-artifact';export_job=a.output/'candidate-export'
        run([sys.executable,'-B',package/'export_mg_roofline.py','--et-source',a.et_source,
             '--gguf',a.gguf,'--reference-deployment',a.reference_deployment,
             '--attention-source-sha256',a.attention_source_sha256,
             '--large-w4a16-policy-file',combined,'--reference-export-job',a.baseline_export_job,
             '--output',artifact,'--job-dir',export_job,'--lock',a.lock],'full-model-candidate-export')
        arms=a.output/'comparison-arms.json'
        arms.write_text(json.dumps(dict(ET_arms=[
            dict(name='ET-baseline',artifact=str(a.baseline_artifact),export_job=str(a.baseline_export_job)),
            dict(name='ET-large',artifact=str(artifact),export_job=str(export_job))]),indent=2)+'\n')
        comparison=a.output/'full-model-comparison'
        run([sys.executable,'-B',package/'run_roofline_default_comparison.py',
             '--ET-SDK-job',a.ET_SDK_job,'--CPP-SDK-job',a.CPP_SDK_job,'--gguf',a.gguf,
             '--arms',arms,'--job',comparison,'--lock',a.lock,'--repetitions','20'],
            'full-model-three-arm-comparison')
        profile=a.output/'candidate-profile'
        run([sys.executable,'-B',package/'profile_roofline_large_candidate.py',
             '--et-source',a.et_source,'--ET-SDK-job',a.ET_SDK_job,
             '--python',sys.executable,'--artifact',artifact,'--after-comparison',comparison,
             '--lock',a.lock,'--job',profile],'full-model-causal-profile')
        state('completed-pending-independent-review',comparison=str(comparison),profile=str(profile))
    except BaseException as e:state('failed',error=str(e));raise


if __name__=='__main__':main()
