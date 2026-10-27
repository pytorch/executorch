"""Test genuine inter-request idle reclamation with retained mini KV/graphs.

No maintenance is inserted between a request's prefill and decode. Both arms
have the same fixed 200ms think time AFTER 64 generated tokens. Cleanup time
and the following request's cold/warm costs remain recorded. This prototype
does not claim a production server idle hook or full-model qualification.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','source-scripts','mg-audit','gemma-audit','output','data-output','lock','python'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);args.data_output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def state(status,**extra):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_models_tested=False,candidate_accepted=False,**extra),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        copied=args.output/'scripts';copied.mkdir()
        for name in ('run_real_mini_four_arms.py','run_allocator_lifecycle.py'):
            shutil.copy2(args.source_scripts/name,copied/name)
        sys.path.insert(0,str(copied))
        from run_real_mini_four_arms import Probe,compare,logits,token
        from run_allocator_lifecycle import DeviceSampler
        audits={}
        for model,path in [('MG',args.mg_audit),('Gemma',args.gemma_audit)]:
            a=json.loads(path.read_text());assert a['model']==model and a['arm']=='baseline' and a['BF16_KV']
            assert a['retained_original_layers']==list(range(4 if model=='MG' else 6))
            for name,meta in a['artifacts'].items():assert sha(Path(a['artifact'])/name)==meta['SHA256']
            audits[model]=a
        manifest=dict(models=audits,scripts={n:sha(copied/n) for n in ('run_real_mini_four_arms.py','run_allocator_lifecycle.py')},
                      output_tokens=64,input_tokens_per_request=1024,bursts=3,concurrencies=[1,16],
                      idle_seconds=.2,maintenance_only_between_complete_requests=True,
                      original_phase_trim_disabled_in_both_arms=True,production_idle_hook_implemented=False)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        state('building-probe-only-idle-operation')
        native=args.output/'native'
        with (args.output/'build.log').open('x') as log:
            subprocess.run([str(args.python),'-B',str(Path(__file__).with_name('build_idle_shared_probe.py')),
                            '--parent',str(args.parent),'--output',str(native)],check=True,stdout=log,stderr=subprocess.STDOUT)
        gate=json.loads((native/'native-gate.json').read_text())
        assert sha(gate['probe'])==gate['probe_SHA256']
        rows=[]
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for stage in ('measure','memory','graph'):
                for model,audit in audits.items():
                    for number in range(1,2 if stage=='graph' else 5):
                        for concurrency in ([16] if stage=='graph' else [1,16]):
                            for arm in (['baseline','allocator'] if number%2 else ['allocator','baseline']):
                                state('running-genuine-idle-mini',stage=stage,model=model,round=number,concurrency=concurrency,arm=arm)
                                directory=args.data_output/stage/model/('round-'+str(number))/('c'+str(concurrency))/arm
                                binary=Path(gate['probe']);profiled=stage=='graph'
                                if profiled:
                                    directory.parent.mkdir(parents=True,exist_ok=True)
                                    wrapper=copied/(model+'-'+arm+'-idle-graph.sh')
                                    assert not wrapper.exists()
                                    wrapper.write_text('#!/bin/sh\nexec /usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=false -o '+str(directory/'graphs')+' '+str(binary)+' "$@"\n')
                                    wrapper.chmod(0o700);binary=wrapper
                                for key in ('ET_CUDA_IDLE_SPARE_BYTES','ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET','CUDA_LAUNCH_BLOCKING'):
                                    os.environ.pop(key,None)
                                os.environ['ET_CUDA_SHARED_IDLE_REFRESH']='1' if arm=='allocator' else '0'
                                probe=Probe(binary,Path(audit['artifact']),directory,False,stage=='memory',profiled)
                                sampler=None;records=[]
                                def request(label,body):
                                    start=time.monotonic_ns();reply=probe.request(body);end=time.monotonic_ns()
                                    records.append(dict(label=label,body=body,begin_ns=start,end_ns=end,reply=reply))
                                    return reply
                                try:
                                    if stage=='memory':sampler=DeviceSampler(directory)
                                    owners=[request('new-owner',dict(op='new'))['seq_id'] for _ in range(concurrency)]
                                    positions={o:0 for o in owners}
                                    for burst in range(3):
                                        begin=time.monotonic_ns();remaining=1024;last=None
                                        while remaining:
                                            width=min(remaining,max(1,audit['prefill_budget']//concurrency));ids=[];tokens=[];select=[]
                                            for owner in owners:
                                                ids += [owner]*width
                                                tokens += [token(owner,x) for x in range(positions[owner],positions[owner]+width)]
                                                positions[owner]+=width;select.append(len(ids)-1)
                                            remaining-=width
                                            last=request('burst-'+str(burst)+'-prefill',dict(op='step',seq_ids=ids,tokens=tokens,select=select,argmax=True,
                                                save_logits=stage=='memory' and number==1 and remaining==0))
                                        first=time.monotonic_ns();streams=[[int(x)] for x in last['argmax']]
                                        assert len(streams)==concurrency
                                        for cycle in range(63):
                                            reply=request('burst-'+str(burst)+'-decode-'+str(cycle),dict(op='step',seq_ids=owners,
                                                tokens=[s[-1] for s in streams],select=list(range(concurrency)),argmax=True,
                                                save_logits=stage=='memory' and number==1 and cycle==62))
                                            assert len(reply['argmax'])==concurrency
                                            for owner,stream,value in zip(owners,streams,reply['argmax']):positions[owner]+=1;stream.append(int(value))
                                        end=time.monotonic_ns()
                                        before=request('burst-'+str(burst)+'-before-idle',dict(op='metrics'))
                                        cleanup_begin=time.monotonic_ns();cleanup=request('burst-'+str(burst)+'-idle-refresh',dict(op='idle_refresh'));cleanup_end=time.monotonic_ns()
                                        deadline=end+200000000
                                        assert cleanup_end<deadline, 'Maintenance exceeded the common think-time budget'
                                        time.sleep(max(0,(deadline-time.monotonic_ns())/1e9))
                                        after=request('burst-'+str(burst)+'-after-idle',dict(op='metrics'))
                                        assert before['snapshot']['live_KV_bytes']==after['snapshot']['live_KV_bytes']
                                        assert cleanup['idle_refresh']['enabled']==(arm=='allocator')
                                        records.append(dict(label='complete-request-group',burst=burst,concurrency=concurrency,
                                            group_wall_seconds=(end-begin)/1e9,first_selected_token_seconds=(first-begin)/1e9,
                                            maintenance_wall_seconds=(cleanup_end-cleanup_begin)/1e9,
                                            idle_budget_seconds=.2,input_count_each=1024,actual_output_count_each=64,
                                            finish_reason='length',output_tokens=streams))
                                    if concurrency==16:
                                        survivor=owners[5]
                                        for owner in owners:
                                            if owner!=survivor:request('release-owner',dict(op='remove',seq_id=owner))
                                        reply=request('automatic-c16-to-c1',dict(op='step',seq_ids=[survivor],tokens=[token(survivor,positions[survivor])],select=[0],argmax=True,save_logits=stage=='memory' and number==1))
                                        assert reply['snapshot']['physical_columns']==2 and reply['snapshot']['live_sequences']==1
                                    rows.append(dict(stage=stage,model=model,round=number,concurrency=concurrency,arm=arm,records=records,directory=str(directory)))
                                finally:
                                    if sampler:sampler.close()
                                    probe.close()
                                (args.output/'records.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        gates=[]
        for stage in ('measure','memory','graph'):
            for model in audits:
                for number in range(1,2 if stage=='graph' else 5):
                    for concurrency in ([16] if stage=='graph' else [1,16]):
                        paired={arm:next(r for r in rows if (r['stage'],r['model'],r['round'],r['concurrency'],r['arm'])==(stage,model,number,concurrency,arm)) for arm in ('baseline','allocator')}
                        a,b=paired['baseline']['records'],paired['allocator']['records']
                        assert len(a)==len(b)
                        for x,y in zip(a,b):
                            assert x['label']==y['label'] and x.get('body')==y.get('body')
                            if x['label']=='complete-request-group':assert x['output_tokens']==y['output_tokens'] and all(len(s)==64 for s in x['output_tokens'])
                            elif stage=='memory' and number==1 and x['reply'].get('logits'):
                                item=compare(logits(x['reply']),logits(y['reply']))
                                assert item['normalized_RMS']==0 and item['max_abs']==0
                                gates.append(dict(model=model,concurrency=concurrency,label=x['label'],**item))
        result=dict(records=rows,numerics=gates,all_input_bodies_and_actual_outputs_identical=True,
                    full_models_tested=False,production_idle_hook_implemented=False,performance_acceptance_pending=True)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('complete-independent-review-required',result_SHA256=sha(args.output/'result.json'))
    except BaseException as error:
        state('failed-preserve-evidence',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
