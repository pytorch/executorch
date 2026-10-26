"""Validate the prefill owner's outputs and four subsequent decode steps.

Run after the unmodified roofline experiment; no validation timing is promoted.
Four fresh processes per ET artifact use the same histories and CUDA graphs.
The two prefill-only performance mixes select no logits, so their vacuous
greedy equality cannot establish full-model candidate correctness.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workflow','ET-SDK-job','et-source','python','lock','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-UTC',required=True)
    p.add_argument('--history-chunk',type=int,choices=[128,256,512,1024],default=512)
    p.add_argument('--require-static-history',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    deadline=datetime.datetime.fromisoformat(a.deadline_UTC);assert deadline.tzinfo is not None
    package=Path(__file__).resolve().parent
    def remaining():
        value=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        assert value>0,'Validation deadline reached; preserve pending outputs'
        return value
    def state(status,**extra):
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            timings_not_performance=True,main_results_changed=False,**extra),indent=2)+'\n')
        pending.replace(a.output/'progress.json')
    try:
        sdk=json.loads((a.ET_SDK_job/'ET-provenance.json').read_text())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
        prefix=subprocess.check_output([str(a.python),'-B','-c',
            'import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        build=a.output/'build';state('building-CPU-only-validation-driver')
        with (a.output/'build.log').open('x') as log:
            for command in [
                ['cmake','-S',str(package/'roofline_validation_project'),'-B',str(build),
                 '-DCMAKE_BUILD_TYPE=Release','-DET_SOURCE='+str(a.et_source),
                 '-DCMAKE_PREFIX_PATH='+sdk['SDK']+';'+prefix],
                ['cmake','--build',str(build),'--parallel','2','--target','roofline_prefill_validation']]:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
        state('waiting-existing-unprofiled-and-profiled-workflow')
        while True:
            remaining();value=json.loads((a.workflow/'progress.json').read_text())
            assert value['status']!='failed',value
            if value['status'] in ['completed-pending-independent-review','comparison-and-profile-completed-awaiting-validation']:break
            os.kill(value['controller_pid'],0);time.sleep(20)
        inputs=json.loads((a.workflow/'full-model-comparison/input-provenance.json').read_text())
        arms=inputs['artifact_arms'];assert {r['name'] for r in arms}=={'ET-baseline','ET-large'}
        binary=build/'roofline_prefill_validation';records=[];env['CUDA_VISIBLE_DEVICES']='0'
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for round_id in [1,2,3,4]:
                for arm in arms:
                    remaining();artifact=Path(arm['artifact'])
                    for name,digest in arm['hashes'].items():assert sha(artifact/name)==digest
                    out=a.output/(arm['name']+'-round'+str(round_id)+'.json')
                    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    state('running-full-model-prefill-validation',round=round_id,arm=arm['name'])
                    with out.with_suffix('.log').open('x') as log:
                        subprocess.run([str(binary),str(artifact/'model.pte'),str(artifact/'aoti_cuda_blob.ptd'),str(out),str(a.history_chunk)],
                            env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
                    data=json.loads(out.read_text());assert data['completed'] and len(data['cases'])==7
                    assert data['history_chunk']==a.history_chunk
                    if a.require_static_history:
                        assert a.history_chunk in [128,256,512]
                        for case in data['cases']:
                            calls={name:count for name,count in case['history_forward_deltas'].items() if count}
                            expected='forward_'+str(a.history_chunk) if arm['name']=='ET-large' else 'forward_others'
                            assert calls=={expected:case['requested_rows']*(data['history_tokens']//a.history_chunk)},calls
                    records.append(dict(round=round_id,arm=arm['name'],file=out.name,SHA256=sha(out)))
                    until=time.monotonic()+60
                    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                        assert time.monotonic()<until;time.sleep(1)
        cases={r['arm']:[] for r in records};comparison=[]
        for r in records:cases[r['arm']].append(json.loads((a.output/r['file']).read_text())['cases'])
        for i,case in enumerate(cases['ET-baseline'][0]):
            all_values=[v[i]['greedy_steps'] for runs in cases.values() for v in runs]
            comparison.append(dict(mix=case['mix'],all_fresh_rounds_and_ET_arms_equal=all(v==all_values[0] for v in all_values)))
        result=dict(records=records,comparison=comparison,binary_SHA256=sha(binary),
            input_provenance=inputs,validation_source_SHA256=sha(package/'roofline_prefill_validation.cpp'),
            selected_prefill_logits_and_four_followup_decode_steps=True,
            history_chunk=a.history_chunk,static_history_execution_verified=a.require_static_history,
            no_performance_measurement_promoted=True,rounds=4)
        (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        state('completed-pending-independent-output-review',all_equal=all(v['all_fresh_rounds_and_ET_arms_equal'] for v in comparison))
    except BaseException as e:state('failed',error=str(e));raise


if __name__=='__main__':main()
