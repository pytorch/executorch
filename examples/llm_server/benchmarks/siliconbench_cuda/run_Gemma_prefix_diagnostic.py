"""Paired same-GPU prefix tests; retain real workloads and add routing controls."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback


def main(job):
    settings = json.loads((job / 'diagnostic-config.json').read_text())
    base = settings['main_configuration']
    completed, failed = [], []
    rc = 1

    def status(state, **extra):
        temp = job / 'progress.pending.json'
        temp.write_text(json.dumps(dict(status=state,completed=completed,failed=failed,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
        temp.replace(job / 'progress.json')

    try:
        if settings.get('wait_for_exit'):
            dependency=Path(settings['wait_for_exit'])
            status('waiting-for-owned-profile-controller',dependency=str(dependency.parent))
            while not dependency.is_file():
                receipt=json.loads((dependency.parent/'launch-receipt.json').read_text())
                proc=Path('/proc')/str(receipt['PID'])
                assert proc.exists(),'Profile controller missing; preserve and inspect its records'
                assert int((proc/'stat').read_text().split(') ',1)[1].split()[19])==receipt['start_ticks']
                time.sleep(30)
        assert base['model'] == 'Gemma' and base['prefix_ON_entries'] == 4
        assert base['active_sessions'] == base['decode_limit'] == 64
        for worker in settings['workers'].values():
            assert hashlib.sha256(Path(worker['path']).read_bytes()).hexdigest() == worker['SHA256']
        status('same-artifact-preflight')
        assert hashlib.sha256(Path(base['PTE']).read_bytes()).hexdigest() == base['PTE_SHA256']
        with Path(base['PTD']).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == base['PTD_SHA256']
        tasks = settings.get('tasks',[(route,split,prefix) for route in ['current','same-route-control']
                 for split in ['chat','agent'] for prefix in ['off','on']])
        for route,split,prefix in tasks:
            name=f'{route}-{split}-{prefix}-c16'
            status('collecting-paired-prefix-performance',current=name)
            cfg=dict(base,split=split,worker=settings['workers'][route]['path'],
                     worker_SHA256=settings['workers'][route]['SHA256'],port=settings['port'])
            cfg['prefix_checkpoint_interval']=settings.get('prefix_checkpoint_interval',base['prefix_checkpoint_interval'])
            configpath=job/(name+'.configuration.json')
            configpath.open('x').write(json.dumps(cfg,indent=2))
            wrapper=job/(name+'.serve.sh')
            wrapper.open('x').write('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.quote(cfg['HTTP_python'])+
                ' -B '+shlex.quote(str(job/'serve_four_opt_main.py'))+' '+shlex.quote(str(configpath))+'\n')
            env=dict(os.environ,ET_MAIN_CONFIG=str(configpath),PYTHONDONTWRITEBYTECODE='1')
            argv=[cfg['HTTP_python'],'-B',str(job/'run_case_64_main.py'),'--wrapper',str(wrapper),
                  '--backend','executorch','--model',cfg['model_id'],'--variant','same-GPU-prefix-diagnostic-'+route,
                  '--prefix-cache',prefix,'--client',str(job/'benchmark_gemma.py'),
                  '--prompts',str(job/(split+'_benchmark_prompts.json')),'--correctness-root',str(job/'not-a-F1-run'),
                  '--python',cfg['HTTP_python'],'--concurrency','16','--port',str(cfg['port']),
                  '--output',str(job/'cases'/name),'--lock',cfg['lock'],'--performance-rounds','4',
                  '--mode','performance','--split',split]
            with (job/(name+'.controller.log')).open('x') as log:
                result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
            (completed if result.returncode==0 else failed).append(name)
            status('paired-case-finished',current=name,returncode=result.returncode)
        rc=1 if failed else 0
        status('performance-complete-hit-theory-and-profile-review-required')
    except BaseException:
        (job/'failure.txt').open('x').write(traceback.format_exc())
        status('failed')
    finally:
        (job/'exit-receipt.json').open('x').write(json.dumps(dict(returncode=rc,completed=completed,failed=failed,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),theoretical_hit_review_pending=True,
            profiling_pending=True),indent=2))
    return rc


if __name__=='__main__':
    raise SystemExit(main(Path(sys.argv[1])))
