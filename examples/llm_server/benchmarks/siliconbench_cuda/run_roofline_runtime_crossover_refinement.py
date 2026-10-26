"""Refine a native cutoff using actual dynamic AOTI, then compare models.

The existing three-arm experiment and profile finish first. Keep the same
Gate/Down policies and all static forwards. Attention's runtime crossover is
measured between its observed256 winner and511 loser, with four-round data.
No main experiment or framework checkout is changed.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['previous-workflow','baseline-export-job','baseline-artifact','ET-SDK-job',
                 'CPP-SDK-job','gguf','et-source','reference-deployment','lock','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--attention-source-sha256',required=True)
    p.add_argument('--deadline-UTC',required=True)
    p.add_argument('--recover-failed-CUDA-export',action='store_true')
    p.add_argument('--workspace-probe-job',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    deadline=datetime.datetime.fromisoformat(a.deadline_UTC);assert deadline.tzinfo is not None
    package=Path(__file__).resolve().parent
    def remaining():
        value=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        assert value>0,'Bounded roofline deadline reached; preserve unfinished outputs'
        return value
    def state(status,**extra):
        f=a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_UTC=a.deadline_UTC,
            main_results_changed=False,model_promoted=False,**extra),indent=2)+'\n')
        f.replace(a.output/'progress.json')
    def run(command,label):
        with (a.output/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state('running-'+label,child_pid=child.pid)
            try:assert child.wait(timeout=remaining())==0,label
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        state('waiting-existing-three-arm-comparison-and-profile')
        while True:
            remaining();s=json.loads((a.previous_workflow/'progress.json').read_text())
            if s['status']=='failed':
                assert a.recover_failed_CUDA_export,s
                assert 'illegal memory access' in (a.previous_workflow/'candidate-export/export.log').read_text()
                assert a.workspace_probe_job is not None
                probe=json.loads((a.workspace_probe_job/'progress.json').read_text())
                assert probe['status']=='completed-pending-independent-memory-safety-review'
                checked={v['arm']:v for v in probe['results']}
                assert checked['old']['exit_code']!=0 and checked['corrected']['exit_code']==0
                assert checked['corrected']['report']['status']=='passed' and checked['corrected']['report']['relative_RMS']<.005
                break
            if s['status']=='completed-pending-independent-review':break
            os.kill(s['controller_pid'],0);time.sleep(20)
        prior=json.loads((a.previous_workflow/'qualified-policies.json').read_text())
        policies=copy.deepcopy(prior['policies']);reports=list(prior['validation_reports'])
        assert len(policies)==len(reports)==3
        attention=policies[2];assert attention['shape']==[6656,4096]
        existing=json.loads(Path(reports[2]).read_text());assert existing['policy']==attention
        cases={v['M']:v for v in existing['cases']}
        assert cases[256]['candidate_over_complete_fallback']<1 and cases[511]['candidate_over_complete_fallback']>1
        original=a.output/'attention-original-policy.json';original.write_text(json.dumps(attention,indent=2)+'\n')
        fine=a.output/'attention-runtime-fine-AOTI'
        rows=[320,384,416,448,464,480,496,504]
        run([sys.executable,'-B',package/'benchmark_roofline_large_w4a16_aoti.py',
             '--et-source',a.et_source,'--gguf',a.gguf,'--policy',original,'--output',fine,
             '--lock',a.lock,'--role','attention_out','--max-M','1024','--additional-M',*rows],
            'attention-runtime-fine-AOTI')
        measured=json.loads((fine/'results.json').read_text())
        assert measured['policy']==attention and measured['packed_SHA256']==attention['packed_SHA256']
        candidates=sorted([v for v in measured['cases'] if 256<=v['M']<=512],key=lambda v:v['M'])
        losers=[v for v in candidates if v['candidate_over_complete_fallback']>=1]
        assert losers,'No runtime crossover reproduced; retain the completed reference and investigate'
        first_loss=losers[0]['M']
        winners=[v for v in candidates if v['M']<first_loss and v['candidate_over_complete_fallback']<1]
        assert winners
        cutoff=max(v['M'] for v in winners)
        # Cut before the first observed runtime regression; retain each earlier
        # measured bucket/config rather than assigning one large-M config everywhere.
        branches=[]
        for item in attention['branches']:
            if branches and branches[-1]['max_M']>=cutoff:break
            branch=copy.deepcopy(item);branch['max_M']=min(branch['max_M'],cutoff)
            branches.append(branch)
        attention['branches']=branches
        selection=dict(changed_role='attention_out',measured_winner_cutoff=cutoff,
            first_observed_runtime_loser=first_loss,all_runtime_rows=measured['cases'],
            static_forward_methods_unchanged=True,Gate_and_Down_policies_unchanged=True)
        (a.output/'runtime-crossover-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        final=a.output/'attention-final-policy.json';final.write_text(json.dumps(attention,indent=2)+'\n')
        if a.recover_failed_CUDA_export:
            # The larger fixed workspace can affect resource use even though
            # valid-row arithmetic is unchanged. Revalidate both reused roles.
            for index,role in [(0,'ffn_down'),(1,'ffn_gate_up')]:
                policy_file=a.output/(role+'-policy.json')
                policy_file.write_text(json.dumps(policies[index],indent=2)+'\n')
                target=a.output/(role+'-dynamic-AOTI')
                run([sys.executable,'-B',package/'benchmark_roofline_large_w4a16_aoti.py',
                     '--et-source',a.et_source,'--gguf',a.gguf,'--policy',policy_file,'--output',target,
                     '--lock',a.lock,'--role',role,'--max-M','1024'],role+'-workspace-revalidation')
                reports[index]=str(target/'results.json')
        validation=a.output/'attention-final-dynamic-AOTI'
        run([sys.executable,'-B',package/'benchmark_roofline_large_w4a16_aoti.py',
             '--et-source',a.et_source,'--gguf',a.gguf,'--policy',final,'--output',validation,
             '--lock',a.lock,'--role','attention_out','--max-M','1024'],
            'attention-final-dynamic-AOTI')
        policies[2]=attention;reports[2]=str(validation/'results.json')
        combined=a.output/'qualified-policies.json';combined.write_text(json.dumps(dict(policies=policies,validation_reports=reports),indent=2)+'\n')
        artifact=a.output/'candidate-artifact';export=a.output/'candidate-export'
        export_command=[sys.executable,'-B',package/'export_mg_roofline.py','--et-source',a.et_source,
             '--gguf',a.gguf,'--reference-deployment',a.reference_deployment,
             '--attention-source-sha256',a.attention_source_sha256,'--large-w4a16-policy-file',combined,
             '--reference-export-job',a.baseline_export_job,'--output',artifact,'--job-dir',export,
             '--lock',a.lock]
        if a.recover_failed_CUDA_export:export_command.extend(['--reference-compile-cache-job',a.previous_workflow/'candidate-export'])
        run(export_command,'full-model-refined-candidate-export')
        arms=a.output/'comparison-arms.json';arms.write_text(json.dumps(dict(ET_arms=[
            dict(name='ET-baseline',artifact=str(a.baseline_artifact),export_job=str(a.baseline_export_job)),
            dict(name='ET-large',artifact=str(artifact),export_job=str(export))]),indent=2)+'\n')
        comparison=a.output/'full-model-comparison'
        run([sys.executable,'-B',package/'run_roofline_default_comparison.py',
             '--ET-SDK-job',a.ET_SDK_job,'--CPP-SDK-job',a.CPP_SDK_job,'--gguf',a.gguf,
             '--arms',arms,'--job',comparison,'--lock',a.lock,'--repetitions','20'],
            'refined-full-model-three-arm-comparison')
        profile=a.output/'candidate-profile'
        run([sys.executable,'-B',package/'profile_roofline_large_candidate.py',
             '--et-source',a.et_source,'--ET-SDK-job',a.ET_SDK_job,'--python',sys.executable,
             '--artifact',artifact,'--after-comparison',comparison,'--lock',a.lock,'--job',profile],
            'refined-full-model-causal-profile')
        state('completed-pending-independent-review',comparison=str(comparison),profile=str(profile),
              selected_attention_cutoff=cutoff,full_output_validation_still_required=True)
    except BaseException as e:state('failed',error=str(e));raise


if __name__=='__main__':main()
