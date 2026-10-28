"""Qualify actual optional idle maintenance through native ServingRuntime."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','overlay','source-scripts','mg-audit','gemma-audit','output','data-output','lock','python'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);args.data_output.mkdir(parents=True,exist_ok=False)
    def state(status,**extra):
        q=args.output/'progress.pending.json';q.write_text(json.dumps(dict(status=status,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_models_tested=False,qualified=False,**extra),indent=2)+'\n');q.replace(args.output/'progress.json')
    try:
        state('verifying-original-mini-artifacts')
        audits={model:json.loads(path.read_text()) for model,path in [('MG',args.mg_audit),('Gemma',args.gemma_audit)]}
        for model,a in audits.items():
            assert a['model']==model and a['arm']=='baseline' and a['BF16_KV']
            for name,m in a['artifacts'].items():assert sha(Path(a['artifact'])/name)==m['SHA256']
        (args.output/'artifact-manifest.json').write_text(json.dumps(audits,indent=2)+'\n')
        state('building-original-and-optional-idle-serving-binaries')
        native=args.output/'native'
        with (args.output/'build.log').open('x') as log:
            subprocess.run([str(args.python),'-B',str(Path(__file__).with_name('build_real_server_idle.py')),'--parent',str(args.parent),'--overlay',str(args.overlay),'--output',str(native)],check=True,stdout=log,stderr=subprocess.STDOUT)
        gate=json.loads((native/'native-gate.json').read_text())
        for name in ('old','new'):assert sha(gate[name]['binary'])==gate[name]['SHA256']
        sys.path.insert(0,str(args.source_scripts))
        from run_allocator_lifecycle import DeviceSampler
        rows=[]
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for stage in ('measure','memory','graph'):
                for model,a in audits.items():
                    for number in range(1,2 if stage=='graph' else 5):
                        for c in ([16] if stage=='graph' else [1,16]):
                            order={1:['old','disabled','enabled'],2:['enabled','disabled','old'],3:['disabled','old','enabled'],4:['enabled','old','disabled']}[number]
                            for arm in order:
                                state('running-real-native-serving-mini',stage=stage,model=model,round=number,concurrency=c,arm=arm)
                                directory=args.data_output/stage/model/('round-'+str(number))/('c'+str(c))/arm;directory.mkdir(parents=True,exist_ok=False)
                                binary=gate['old' if arm=='old' else 'new']['binary']
                                argv=[binary,'--pte='+str(Path(a['artifact'])/'model.pte'),'--data_path='+str(Path(a['artifact'])/'aoti_cuda_blob.ptd'),'--concurrency='+str(c),'--budget='+str(a['prefill_budget']),'--shared_idle_refresh='+('true' if arm=='enabled' else 'false')]
                                if stage=='graph':argv=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','-o',str(directory/'graphs'),*argv]
                                env=dict(os.environ)
                                for key in ('ET_CUDA_IDLE_SPARE_BYTES','ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET','CUDA_LAUNCH_BLOCKING'):env.pop(key,None)
                                env['ET_CUDA_SHARED_IDLE_REFRESH']='1' if arm=='enabled' else '0'
                                sampler=DeviceSampler(directory) if stage=='memory' else None
                                try:
                                    with (directory/'stdout.jsonl').open('x') as out,(directory/'stderr.log').open('x') as err:
                                        completed=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180)
                                    assert completed.returncode==0,(directory,completed.returncode)
                                    lines=(directory/'stdout.jsonl').read_text().splitlines()
                                    record=json.loads(next(x for x in reversed(lines) if x.startswith('{')))
                                    assert not record.get('failed') and record['native_serving_API']
                                    assert record['new_runtime']==(arm!='old') and record['idle_refresh_enabled']==(arm=='enabled')
                                    assert len(record['groups'])==3
                                    rows.append(dict(stage=stage,model=model,round=number,concurrency=c,arm=arm,directory=str(directory),result=record))
                                finally:
                                    if sampler:sampler.close()
                                (args.output/'records.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        paired=0
        for stage in ('measure','memory','graph'):
            for model in audits:
                for n in range(1,2 if stage=='graph' else 5):
                    for c in ([16] if stage=='graph' else [1,16]):
                        arms={a:next(r for r in rows if (r['stage'],r['model'],r['round'],r['concurrency'],r['arm'])==(stage,model,n,c,a))['result'] for a in ('old','disabled','enabled')}
                        for i in range(3):
                            old=arms['old']['groups'][i]
                            assert len(old['output_tokens'])==c and all(len(s)==64 for s in old['output_tokens'])
                            for a in ('disabled','enabled'):
                                new=arms[a]['groups'][i]
                                assert old['input_tokens']==new['input_tokens'] and old['output_tokens']==new['output_tokens']
                                assert [x['prefilled_prompt_tokens'] for x in old['requests']]==[x['prefilled_prompt_tokens'] for x in new['requests']]
                            paired+=1
        result=dict(records=rows,paired_original_disabled_enabled_groups=paired,all_actual_input_and_output_tokens_equal=True,
                    original_SDK_modified=False,full_models_tested=False,performance_acceptance_pending=True,
                    deterministic_diagnostic_decoder=True,HTTP_pipeline_tested=False)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('complete-independent-review-required',result_SHA256=sha(args.output/'result.json'))
    except BaseException as e:state('failed-preserve-evidence',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
