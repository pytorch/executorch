"""Collect single-round chat/agent matrices, then the selected quality checks.

Configuration names exact stock launch wrappers and existing public fixtures.
Each result has 100 measured requests and three unmeasured warmup requests.
Previous experiment directories are never overwritten.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    completed=[]
    started=time.monotonic()

    def state(status,**extra):
        value=dict(status=status,controller_pid=os.getpid(),
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   elapsed_s=time.monotonic()-started,completed=completed,
                   accepted=False,**extra)
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(value,indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    def run(name,command,timeout):
        (args.output/(name+'-command.json')).write_text(json.dumps(dict(argv=list(map(str,command))),indent=2)+'\n')
        with (args.output/(name+'.log')).open('x') as log:
            subprocess.run(list(map(str,command)),stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=timeout)

    try:
        state('waiting-dependencies-or-export')
        dependency=cfg.get('after_export_job')
        if dependency:
            root=Path(dependency)
            while True:
                progress=root/'progress.json'
                if not progress.is_file():
                    launch=json.loads(Path(cfg['after_export_launch']).read_text())
                    proc=Path('/proc')/str(launch['PID'])/'cmdline'
                    assert proc.is_file() and str(root.parent).encode() in proc.read_bytes()
                    time.sleep(10)
                    continue
                record=json.loads(progress.read_text())
                assert record['status']!='failed',record
                if record['status'] in ('full-artifact-ready-pending-full-model-qualification','artifact-exported-and-audited'):break
                proc=Path('/proc')/str(record['controller_pid'])/'cmdline'
                assert proc.is_file() and str(root.parent).encode() in proc.read_bytes()
                state('waiting-confirmed-live-export',export_PID=record['controller_pid'],export_status=record['status'])
                time.sleep(20)
        tasks=cfg['tasks']
        assert tasks
        for task in tasks:
            for filename in ('wrapper','client','inputs'):
                assert Path(task[filename]).exists(),(task['name'],filename)
            if task.get('artifact_audit'):
                audit=json.loads(Path(task['artifact_audit']).read_text())
                assert audit['independent_capacity'] and audit['initial_runtime_KV']==512
                assert audit['metadata']['get_max_prefill_chunk']==task['prefill_budget']
                assert audit['logical_context']==131072 and audit['selected_logit_limit']==64
                for filename,record in audit['artifacts'].items():
                    assert sha(Path(task['artifact_audit']).parent/filename)==record['sha256']
            task['wrapper_SHA256']=sha(task['wrapper'])
            task['client_SHA256']=sha(task['client'])
        (args.output/'configuration.json').write_text(json.dumps(dict(config=cfg,config_SHA256=sha(args.config),
            driver_SHA256=sha(__file__),runner_SHA256=sha(package/'run_case_24h.py'),
            performance_rounds=1,discarded_rounds=[],repetition_SD=None),indent=2)+'\n')
        for mode in ('performance','correctness'):
            for task in tasks:
                if mode=='correctness' and task.get('quality_policy')=='reuse':continue
                for split in (('chat','agent') if mode=='performance' else ('agent',)):
                    for prefix in ('off','on'):
                        for c in ((1,2,4,8,16) if mode=='performance' else (16,1,2,4,8)):
                            name=f"{task['name']}-{mode}-{split}-{prefix}-c{c}"
                            case=args.output/name
                            command=[task['benchmark_python'],'-B',package/'run_case_24h.py',
                                '--wrapper',task['wrapper'],'--backend',task['backend'],'--model',task['model'],
                                '--variant','default-memory-matched-prefill-single-round',
                                '--prefix-cache',prefix,'--client',task['client'],
                                '--prompts',Path(task['inputs'])/(split+'_benchmark_prompts.json'),
                                '--correctness-root',Path(task['inputs'])/'correctness','--python',task['benchmark_python'],
                                '--concurrency',c,'--port',cfg['port'],'--output',case,'--lock',cfg['lock'],
                                '--mode',mode,'--split',split,'--performance-rounds','1']
                            if mode=='correctness':
                                scope='full' if c in (1,16) and task.get('quality_policy')=='full' else 'smoke'
                                command+=['--quality-scope',scope,'--correctness-max-tokens','1024',
                                    '--correctness-chat-template-kwargs',json.dumps(dict(enable_thinking=False,
                                    reasoning_strength='low',current_date='2026-10-21'))]
                            state('collecting-'+mode,current_case=str(case),task=task['name'])
                            run(name,command,43200)
                            done=json.loads((case/'progress.json').read_text())
                            assert done['status']=='completed',done
                            if mode=='performance':
                                values=json.loads((case/'performance-aggregate.json').read_text())
                                assert values['successful']['values']==[100] and values['failed']['values']==[0]
                                assert all(value['sample_stddev'] is None for value in values.values())
                            completed.append(name)
            if mode=='performance':
                (args.output/'performance-result.json').write_text(json.dumps(dict(completed=completed,
                    performance_cells=len(completed),raw_review_pending=True,full_quality_pending=True),indent=2)+'\n')
                state('performance-complete-starting-deferred-quality')
        (args.output/'result.json').write_text(json.dumps(dict(completed=completed,independent_raw_review_pending=True),indent=2)+'\n')
        state('single-round-matrix-and-deferred-quality-collected')
    except BaseException as error:
        state('failed-preserve-existing-results',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
