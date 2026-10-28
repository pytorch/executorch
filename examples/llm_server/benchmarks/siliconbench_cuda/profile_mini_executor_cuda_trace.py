"""Collect actual CUDA kernel and API timing for the fixed-length mini diagnostic."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import traceback


def main(args):
    p=args.job
    assert p.is_dir() and not (p/'cases').exists()
    rc=1
    try:
        assert json.loads((args.executor_job/'exit-receipt.json').read_text())['returncode']==0
        nsys=shutil.which('nsys');assert nsys
        sdk=args.base/'MG-mini-required-side-binding-native-release-20261027-v2/native/executorch'
        artifact=args.base/'four-opt-MG-mini81-stable-layout128-qualification-20261028-v1/same-PTE'
        env=dict(os.environ,LD_LIBRARY_PATH=str(sdk/'cmake-out/lib'),
            ET_CUDA_RING_JOURNAL_BUFFERS='0',ET_CUDA_RING_ARCHIVE_BATCH_ROWS='64',ET_CUDA_SHARED_IDLE_REFRESH='1')
        rows=[]
        with (args.base/'gpu-benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for c in [1,64]:
                folder=p/'cases'/str(c);folder.mkdir(parents=True)
                cmd=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none',
                    '--force-overwrite=false','-o',str(folder/'cuda-trace'),
                    str(args.executor_job/'executor-mini'),'--pte='+str(artifact/'model.pte'),
                    '--data_path='+str(artifact/'aoti_cuda_blob.ptd'),'--concurrency='+str(c),
                    '--budget=4096','--prompt_tokens=1024','--prefix_entries=0','--cold_sessions=true',
                    '--admission_quiet_us=300','--admission_maximum_us=2000',
                    '--diagnostic_admission_barrier=false','--shared_idle_refresh=false']
                (folder/'command.json').write_text(json.dumps(cmd,indent=2))
                with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
                    subprocess.run(cmd,stdout=out,stderr=err,env=env,check=True,timeout=600)
                cmd=[nsys,'stats','--report=cuda_gpu_kern_sum,cuda_api_sum','--format=csv',str(folder/'cuda-trace.nsys-rep')]
                with (folder/'stats.csv').open('x') as out,(folder/'stats.stderr.log').open('x') as err:
                    subprocess.run(cmd,stdout=out,stderr=err,env=env,check=True,timeout=300)
                db=sqlite3.connect(folder/'cuda-trace.sqlite')
                tables=[x[0] for x in db.execute("select name from sqlite_master where type='table'")]
                kernels=[]
                if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                    columns=[x[1] for x in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
                    name='shortName' if 'shortName' in columns else 'demangledName'
                    kernels=[dict(name=n,calls=k,total_ns=t,mean_ns=a) for n,k,t,a in db.execute(
                        'select s.value,count(*),sum(k.end-k.start),avg(k.end-k.start) '
                        'from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.'+name+' '
                        'group by s.value order by sum(k.end-k.start) desc')]
                rows.append(dict(c=c,tables=tables,kernels=kernels,directory=str(folder)))
                (p/'progress.json').write_text(json.dumps(dict(completed=[x['c'] for x in rows],mini_only=True)))
        (p/'result.json').write_text(json.dumps(dict(rows=rows,mini_only=True,
            traced_timings_not_main_performance=True,full_main_unchanged=True),indent=2))
        rc=0
    except BaseException:
        (p/'failure.txt').write_text(traceback.format_exc())
    finally:
        (p/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['job','base','executor-job']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
