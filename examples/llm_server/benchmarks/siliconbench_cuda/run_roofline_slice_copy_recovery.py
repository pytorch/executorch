"""Recover complete roofline export while retaining nested slice-copy semantics.

Reuse validated policies and unchanged static delegates from the preserved
terminal export. Dynamic delegates are freshly compiled. Full three-arm timing,
causal profiling and independent prefill/decode output checks remain required.
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
    p.add_argument('--ownership-probe-job',type=Path,required=True)
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
        state('checking-preserved-missing-fallback-export')
        s=json.loads((a.previous_workflow/'progress.json').read_text())
        assert s['status']=='failed'
        assert 'Method forward_others missing fallback kernels' in (a.previous_workflow/'candidate-export/export.log').read_text()
        probe=json.loads((a.ownership_probe_job/'results.json').read_text())
        assert probe['status']=='passed' and probe['all_seven_preserved_static_paths_exist']
        assert len(probe['cases'])==6 and all(v['bitwise_equal'] and v['input_unchanged_after_output_mutation'] for v in probe['cases'])
        prior=json.loads((a.previous_workflow/'qualified-policies.json').read_text())
        assert len(prior['policies'])==len(prior['validation_reports'])==3
        for policy,report in zip(prior['policies'],prior['validation_reports']):
            checked=json.loads(Path(report).read_text())
            assert checked['policy']==policy
        combined=a.output/'qualified-policies.json'
        combined.write_text(json.dumps(prior,indent=2)+'\n')
        cutoff=prior['policies'][2]['branches'][-1]['max_M']
        previous_selection=a.previous_workflow/'runtime-crossover-selection.json'
        (a.output/'runtime-crossover-selection.json').write_text(previous_selection.read_text())
        artifact=a.output/'candidate-artifact';export=a.output/'candidate-export'
        export_command=[sys.executable,'-B',package/'export_mg_roofline.py','--et-source',a.et_source,
             '--gguf',a.gguf,'--reference-deployment',a.reference_deployment,
             '--attention-source-sha256',a.attention_source_sha256,'--large-w4a16-policy-file',combined,
             '--reference-export-job',a.baseline_export_job,'--output',artifact,'--job-dir',export,
             '--lock',a.lock]
        export_command.extend(['--slice-copy-recovery-job',a.previous_workflow/'candidate-export'])
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
