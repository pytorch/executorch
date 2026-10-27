# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Compare both ring implementations with the same compilation and graph mode."""
import argparse, datetime, fcntl, hashlib, json, os, subprocess, sys, traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pkg=Path(__file__).resolve().parent
    manifest=json.loads((pkg/'source-manifest.json').read_text())
    for group in ['sources','native_candidate_sources']:
        for name,expected in manifest[group].items():
            assert hashlib.sha256((pkg/name).read_bytes()).hexdigest()==expected,name
    completed=[]
    def state(status,**kw):
        v=dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=os.getpid(),completed=completed,ET_mini_model_qualified=False,**kw)
        pending=a.output/'progress.pending.json';pending.write_text(json.dumps(v,indent=2)+'\n');pending.replace(a.output/'progress.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('CUDA_LAUNCH_BLOCKING',None)
    try:
        state('waiting-for-exclusive-GPU')
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for mode in ['compiled','eager']:
                for model in ['Gemma','MG']:
                    case=a.output/(model+'-'+mode)
                    cache=a.output/(model+'-'+mode+'-inductor-cache')
                    case_env=dict(env,TORCHINDUCTOR_CACHE_DIR=str(cache),TRITON_CACHE_DIR=str(a.output/(model+'-'+mode+'-triton-cache')))
                    argv=[sys.executable,'-B','-u',str(pkg/'probe_ring_journal_compile.py'),'--model',model,'--baseline',str(pkg/('baseline_'+model+'_sdpa.py')),'--output',str(case),'--baseline-execution',mode,'--candidate-execution',mode]
                    state('running-balanced-ring-comparison',case=str(case),argv=argv)
                    with (a.output/(case.name+'.log')).open('x') as log:
                        subprocess.run(argv,env=case_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
                    result=json.loads((case/'result.json').read_text())
                    assert result['kernel_numerical_gate_passed'] and result['graph_dynamic_metadata_qualified']
                    assert result['compiled_journal_gate']==(mode=='compiled')
                    generated=[]
                    for code in cache.rglob('*.py'):
                        text=code.read_text(errors='replace')
                        if 'def call(' in text:
                            generated.append(dict(path=str(code),SHA256=hashlib.sha256(code.read_bytes()).hexdigest(),call_body=text[text.index('def call('):]))
                    (case/'generated-call-audit.json').write_text(json.dumps(generated,indent=2)+'\n')
                    completed.append(case.name)
            state('balanced-kernel-comparison-collected',mini_native_four_arm_gate_pending=True)
    except BaseException as e:
        state('failed-preserve-evidence',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
