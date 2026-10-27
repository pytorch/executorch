"""Run complete destination F1 after the native performance matrix finishes.

The original matrix supplies balanced c2/c4/c8 smoke checks. This additional
controller verifies all 1146 items for both shots at c1/c16, for each model and
prefix setting. It does not repeat performance or change framework sources.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    completed=[]
    def state(status,**extra):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed,
            independent_raw_review_pending=True,**extra),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        while not (args.native_job/'matrix/result.json').is_file():
            state('waiting-performance-and-smoke-completion',dependency=str(args.native_job))
            for relative in ('progress.json','matrix/progress.json'):
                path=args.native_job/relative
                if path.exists():
                    record=json.loads(path.read_text())
                    assert not record['status'].startswith('failed'),record
            time.sleep(30)
        cfg=json.loads((args.native_job/'matrix-config.json').read_text())
        assert len(cfg['tasks'])==2
        for task in cfg['tasks']:
            for prefix in ('off','on'):
                for c in (16,1):
                    name=f"{task['name']}-correctness-agent-{prefix}-c{c}-full"
                    case=args.output/name
                    command=[task['benchmark_python'],'-B',package/'run_case_24h.py',
                        '--wrapper',task['wrapper'],'--backend',task['backend'],'--model',task['model'],
                        '--variant','default-memory-matched-prefill-single-round','--prefix-cache',prefix,
                        '--client',task['client'],'--prompts',Path(task['inputs'])/'agent_benchmark_prompts.json',
                        '--correctness-root',Path(task['inputs'])/'correctness','--python',task['benchmark_python'],
                        '--concurrency',str(c),'--port',str(cfg['port']),'--output',case,'--lock',cfg['lock'],
                        '--mode','correctness','--split','agent','--quality-scope','full',
                        '--correctness-max-tokens','1024','--correctness-chat-template-kwargs',
                        json.dumps(dict(enable_thinking=False,reasoning_strength='low',current_date='2026-10-21'))]
                    (args.output/(name+'-command.json')).write_text(json.dumps(dict(argv=list(map(str,command))),indent=2)+'\n')
                    state('collecting-deferred-complete-F1',current_case=str(case))
                    with (args.output/(name+'.log')).open('x') as log:
                        subprocess.run(list(map(str,command)),stdin=subprocess.DEVNULL,stdout=log,
                            stderr=subprocess.STDOUT,check=True,timeout=43200)
                    assert json.loads((case/'progress.json').read_text())['status']=='completed'
                    scores=json.loads((case/'correctness-scores.json').read_text())
                    assert len(scores)==2 and {s['shots'] for s in scores}=={0,5}
                    assert all(s['n_scored']==1146 and s['request_errors']==0 and
                        s['n_missing_responses']==s['n_extra_responses']==0 for s in scores)
                    completed.append(name)
        assert len(completed)==8
        (args.output/'result.json').write_text(json.dumps(dict(completed=completed,
            independent_raw_review_pending=True),indent=2)+'\n')
        state('deferred-complete-F1-collected')
    except BaseException as error:
        state('failed-preserve-existing-results',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
