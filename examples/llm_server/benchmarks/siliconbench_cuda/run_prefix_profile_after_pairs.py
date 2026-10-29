"""After paired measurements, compute hit theory and profile cache/control waits."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import time
import traceback


def main(args):
    job,pairs=args.job,args.pairs
    completed,failed=[],[]
    rc=1

    def status(state,**extra):
        temp=job/'progress.pending.json'
        temp.write_text(json.dumps(dict(status=state,completed=completed,failed=failed,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
        temp.replace(job/'progress.json')

    try:
        status('waiting-for-owned-paired-measurements')
        while not (pairs/'exit-receipt.json').is_file():
            receipt=json.loads((pairs/'launch-receipt.json').read_text())
            proc=Path('/proc')/str(receipt['PID'])
            assert proc.exists(),'Paired controller disappeared; inspect original records'
            assert int((proc/'stat').read_text().split(') ',1)[1].split()[19])==receipt['start_ticks']
            time.sleep(30)
        receipt=json.loads((pairs/'exit-receipt.json').read_text())
        assert receipt['returncode']==0 and len(receipt['completed'])==8 and not receipt['failed']
        cfg=json.loads((pairs/'diagnostic-config.json').read_text())['main_configuration']
        status('computing-exact-token-theoretical-prefixes')
        with (job/'theoretical-hit-controller.log').open('x') as log:
            subprocess.run([cfg['HTTP_python'],'-B',str(job/'analyze_prefix_theoretical_hits.py'),
                '--job',str(pairs)],stdout=log,stderr=subprocess.STDOUT,check=True)
        binding=json.loads((job/'worker-binding.json').read_text())
        assert hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest()==binding['SHA256']
        for prefix in ['off','on']:
            name='same-route-agent-'+prefix+'-c16'
            status('profiling-control-prefix-and-CUDA',current=name)
            case=job/name;case.mkdir()
            profilecfg=dict(cfg,worker=binding['path'],worker_SHA256=binding['SHA256'],split='agent',port=20613)
            configpath=case/'configuration.json';configpath.open('x').write(json.dumps(profilecfg,indent=2))
            wrapper=case/'serve.sh'
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--trace-fork-before-exec=true','--wait=all','--cuda-graph-trace=node',
                '--stats=false','--force-overwrite=false','--output='+str(case/'timeline'),
                cfg['HTTP_python'],'-B',str(pairs/'serve_four_opt_main.py'),str(configpath)]
            wrapper.open('x').write('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(command)+'\n')
            env=dict(os.environ,ET_MAIN_CONFIG=str(configpath),PYTHONDONTWRITEBYTECODE='1')
            argv=[cfg['HTTP_python'],'-B',str(pairs/'run_case_64_main.py'),'--wrapper',str(wrapper),
                '--backend','executorch','--model',cfg['model_id'],'--variant','prefix-observation-only-not-performance-measurement',
                '--prefix-cache',prefix,'--client',str(pairs/'benchmark_gemma.py'),
                '--prompts',str(pairs/'agent_benchmark_prompts.json'),'--correctness-root',str(job/'not-F1'),
                '--python',cfg['HTTP_python'],'--concurrency','16','--port','20613','--output',str(case/'HTTP'),
                '--lock',cfg['lock'],'--performance-rounds','1','--mode','performance','--split','agent']
            with (case/'controller.log').open('x') as log:
                run=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=9000)
            traces=list(case.glob('*.nsys-rep'))
            if run.returncode==0:
                completed.append(name)
            else:
                failed.append(dict(case=name,returncode=run.returncode))
            if traces:
                with (case/'nsys-summary.csv').open('x') as out,(case/'nsys-stats.log').open('x') as err:
                    summary=subprocess.run(['nsys','stats','--report','cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum',
                        '--format','csv','--force-export=true',str(traces[0])],stdout=out,stderr=err)
                (case/'profile-receipt.json').open('x').write(json.dumps(dict(HTTP_returncode=run.returncode,
                    nsys_stats_returncode=summary.returncode,traces=[str(p) for p in traces],
                    instrumented_and_profiled_times_not_used_as_perf_results=True),indent=2))
            else:
                (case/'profile-receipt.json').open('x').write(json.dumps(dict(HTTP_returncode=run.returncode,
                    trace_missing=True,CPU_prefix_control_events_still_available=True),indent=2))
        rc=1 if failed else 0
        status('profiles-complete-independent-bottleneck-review-required')
    except BaseException:
        (job/'failure.txt').open('x').write(traceback.format_exc())
        status('failed')
    finally:
        (job/'exit-receipt.json').open('x').write(json.dumps(dict(returncode=rc,completed=completed,failed=failed,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_review_pending=True),indent=2))
    return rc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
