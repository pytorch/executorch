# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Run a balanced compiled kernel gate, then real mini export and four arms."""
import argparse, datetime, fcntl, json, os, subprocess, sys, traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['native-preparation','baseline-audit','gguf','output','lock']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model',choices=['MG','Gemma'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pkg=Path(__file__).resolve().parent
    def state(status,**extra):
        temp=a.output/'progress.pending.json';temp.write_text(json.dumps(dict(status=status,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),real_mini_qualified=False,full_models_paused=True,**extra),indent=2)+'\n');temp.replace(a.output/'progress.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_CACHE_DIR=str(a.output/'kernel-inductor-cache'),TRITON_CACHE_DIR=str(a.output/'kernel-triton-cache'))
    env.pop('CUDA_LAUNCH_BLOCKING',None)
    try:
        state('checking-balanced-compiled-kernel')
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            argv=[sys.executable,'-B','-u',str(pkg/'probe_ring_journal_compile.py'),'--model',a.model,'--baseline',str(pkg/('baseline_'+a.model+'_sdpa.py')),'--output',str(a.output/'compiled-probe'),'--baseline-execution','compiled','--candidate-execution','compiled']
            with (a.output/'compiled-probe.log').open('x') as log:
                subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        state('exporting-and-measuring-real-mini')
        argv=[sys.executable,'-B','-u',str(pkg/'recover_real_mini_export.py'),'--model',a.model,'--native-preparation',str(a.native_preparation),'--baseline-audit',str(a.baseline_audit),'--compiled-probe',str(a.output/'compiled-probe/result.json'),'--gguf',str(a.gguf),'--output',str(a.output/'recovery'),'--lock',str(a.lock)]
        with (a.output/'recovery.log').open('x') as log:
            subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=21600)
        state('four-arm-data-collected-review-required',performance_acceptance_pending=True)
    except BaseException as e:
        state('failed-preserve-evidence',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
