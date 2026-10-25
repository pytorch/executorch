"""Apply the low-peak migration overlay to a terminal isolated core build.

Preserve prior sources, binaries, logs and SHA receipts. Compile with CUDA
hidden; selected GPU ownership/history/rollback tests run as a separate child
while this CPU parent holds the shared lock. Model HTTP qualification follows.
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
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['base-job','output','python','lock']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((a.base_job/'progress.json').read_text())
    assert previous['status']=='CPU-core-build-completed'
    assert not Path('/proc',str(previous['controller_pid'])).exists()
    source=Path(previous['source']);build=source/'cmake-out'
    package=Path(__file__).resolve().parent
    patch=package/'gemma_runtime_columns_lowpeak.patch'
    spec=json.loads((package/'gemma_runtime_columns_lowpeak.json').read_text())
    assert sha(patch)==spec['patch_SHA256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDAARCHS='120',TORCH_CUDA_ARCH_LIST='12.0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MAX_JOBS='2')
    env['PATH']=str(a.python.parent)+os.pathsep+env['PATH']
    def state(status,**extra):
        (a.output/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),model_HTTP_qualified=False,main_results_changed=False,**extra),indent=2)+'\n')
    def run(command,timeout=3600):
        with (a.output/'build.log').open('a') as log:
            log.write(json.dumps([str(v) for v in command])+'\n');log.flush()
            subprocess.run([str(v) for v in command],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        state('preserving-terminal-isolated-core')
        old={}
        for name,record in spec['files'].items():
            assert sha(source/name)==record['before_SHA256'],name
            dest=a.output/'before-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
            old[name]=sha(dest)
        binary=build/'backends/cuda/test_cuda_kv_cache';dest=a.output/'before-test_cuda_kv_cache'
        shutil.copy2(binary,dest);old['test_binary']=sha(dest)
        (a.output/'before-provenance.json').write_text(json.dumps(dict(base_progress=previous,base_progress_SHA256=sha(a.base_job/'progress.json'),source_and_test_SHA256=old),indent=2)+'\n')
        run(['git','-C',source,'apply','--check',patch.resolve()]);run(['git','-C',source,'apply',patch.resolve()])
        for name,record in spec['files'].items():assert sha(source/name)==record['after_SHA256'],name
        state('building-lowpeak-isolated-core',GPU_work=False,patch_SHA256=sha(patch))
        run(['cmake','--build',build,'--parallel','2','--target','test_cuda_kv_cache','install'],7200)
        state('waiting-GPU-lock-for-KV-tests',GPU_work=False)
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            testenv=dict(env,CUDA_VISIBLE_DEVICES='0')
            command=[str(binary),'--gtest_filter=CudaRuntimeColumnCacheTest.*:CudaKVPoolTest.*:CudaBatchedSequenceCacheTest.*','--gtest_output=json:'+str(a.output/'GPU-tests.json')]
            with (a.output/'GPU-tests.log').open('x') as log:
                child=subprocess.Popen(command,env=testenv,stdout=log,stderr=subprocess.STDOUT)
                state('running-KV-history-Gemma-geometry-and-rollback-tests',child_pid=child.pid)
                assert child.wait(timeout=900)==0,'Inspect preserved GPU-tests.log'
            tests=json.loads((a.output/'GPU-tests.json').read_text());assert tests['failures']==0 and tests['disabled']==0
            allnames=[c['name'] for suite in tests['testsuites'] for c in suite['testsuite']]
            assert 'GemmaGeometryMigratesEveryOwnerAtRuntimeColumnWidths' in allnames
            assert 'FailedColumnGrowthRestoresOwnersAndRecapturesMovedBindings' in allnames
            deadline=time.monotonic()+60
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<deadline;time.sleep(1)
        (a.output/'qualified-core.json').write_text(json.dumps(dict(source=str(source),patch_SHA256=sha(patch),source_files={name:sha(source/name) for name in spec['files']},test_binary_SHA256=sha(binary),GPU_test_report_SHA256=sha(a.output/'GPU-tests.json'),test_count=tests['tests'],model_HTTP_qualified=False),indent=2)+'\n')
        state('core-KV-tests-completed-pending-Gemma-HTTP',test_count=tests['tests'])
    except BaseException as error:
        (a.output/'traceback.txt').write_text(traceback.format_exc());state('failed',error=str(error));raise


if __name__=='__main__':main()
