"""Profile model/launch work after the unprofiled roofline comparison.

One existing API forward per batch mix, with NVTX ranges from the same input
driver. Profile timing stays separate from performance averages.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['et-source','ET-SDK-job','python','artifact','after-comparison','lock','job']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.job.mkdir(parents=True,exist_ok=False)
    def state(status,**fields):
        (args.job/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile_not_performance_data=True,**fields),indent=2)+'\n')
    try:
        nsys=shutil.which('nsys');assert nsys,'Nsight Systems is required for this separate trace'
        package=Path(__file__).resolve().parent
        sdk=json.loads((args.ET_SDK_job/'ET-provenance.json').read_text())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MAX_JOBS='2')
        prefix=subprocess.check_output([str(args.python),'-B','-c','import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        build=args.job/'profile-build';state('building-isolated-profile-driver')
        commands=[['cmake','-S',str(package/'roofline_profile_project'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DET_SOURCE='+str(args.et_source),'-DCMAKE_PREFIX_PATH='+sdk['SDK']+';'+prefix],['cmake','--build',str(build),'--parallel','2','--target','roofline_et_profile']]
        with (args.job/'build.log').open('a') as log:
            for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        state('waiting-unprofiled-comparison')
        deadline=time.monotonic()+21600
        while True:
            path=args.after_comparison/'progress.json'
            if path.exists():
                progress=json.loads(path.read_text());assert progress['status']!='failed'
                if progress['status']=='completed-pending-independent-review':break
            assert time.monotonic()<deadline;time.sleep(20)
        binary=build/'roofline_et_profile';out=args.job/'profile-driver.json';trace=args.job/'candidate'
        env['CUDA_VISIBLE_DEVICES']='0'
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            teardown=time.monotonic()+120
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<teardown;time.sleep(1)
            state('profiling-existing-large-M-artifact')
            command=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(trace),str(binary),str(args.artifact/'model.pte'),str(args.artifact/'aoti_cuda_blob.ptd'),str(out),'4']
            with (args.job/'nsys.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            data=json.loads(out.read_text());assert data['completed'] and len(data['cases'])==7
            assert all(c['same_prefix_greedy_repeats_equal'] and sum(c['actual_method_call_deltas'].values())==7 for c in data['cases'])
            for mix in ['p512','d8p512']:
                case=next(c for c in data['cases'] if c['mix']==mix)
                assert case['actual_method_call_deltas']['forward_others']==7
            with (args.job/'export.log').open('x') as log:subprocess.run([nsys,'export','--type=sqlite','--output='+str(trace.with_suffix('.sqlite')),str(trace.with_suffix('.nsys-rep'))],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            teardown=time.monotonic()+120
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<teardown;time.sleep(1)
        (args.job/'profile-provenance.json').write_text(json.dumps(dict(binary_SHA256=sha(binary),PTE_SHA256=sha(args.artifact/'model.pte'),driver_SHA256=sha(out),SQLite_SHA256=sha(trace.with_suffix('.sqlite')),nsys_command=command,one_existing_forward_per_mix_verified=True,profile_timing_not_used_for_performance_averages=True),indent=2)+'\n')
        state('completed-pending-causal-trace-review')
    except BaseException as error:state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
