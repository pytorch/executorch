"""Run existing official API drivers with stock default CUDA llama.cpp.

ET artifact arms are explicit inputs; no framework build or kernel patch is
performed here. All arms run on one idle GPU, four fresh process rounds with
the first discarded. This validates roofline inputs, not HTTP task F1.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(16*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['ET-SDK-job','CPP-SDK-job','gguf','arms','job','lock']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=20)
    a=p.parse_args();a.job.mkdir(parents=True,exist_ok=False)
    def state(status,**fields):
        (a.job/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),indent=2)+'\n')
    def wait_ready(path,ready):
        start=time.monotonic()
        while True:
            if path.exists():
                value=json.loads(path.read_text())
                assert value.get('status')!='failed' and value.get('phases',{}).get('controller')!='failed',str(path)
                if ready(value):return value
            assert time.monotonic()-start<21600,'Input preparation remains incomplete'
            time.sleep(20)
    def teardown():
        start=time.monotonic()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            assert time.monotonic()-start<90,'Wait for owned CUDA context teardown';time.sleep(1)
    try:
        arms=json.loads(a.arms.read_text())['ET_arms'];assert arms and len({x['name'] for x in arms})==len(arms)
        assert all(x['name'].startswith('ET-') and all(c.isalnum() or c=='-' for c in x['name']) for x in arms)
        assert digest(a.gguf)=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
        state('waiting-native-SDK-and-exports')
        et=wait_ready(a.ET_SDK_job/'progress.json',lambda x:x.get('phases',{}).get('ET')=='ready')
        # The CPP provenance is independently complete even if a prior ET
        # preparation in its controller failed. Verify the stock binary itself.
        start=time.monotonic();cpp_prov=a.CPP_SDK_job/'CPP-provenance.json'
        while not cpp_prov.exists():
            assert time.monotonic()-start<7200;time.sleep(20)
        cp=json.loads(cpp_prov.read_text());assert cp['CUDA'] and cp['FORCE_CUBLAS'] is False and cp['source_unmodified']
        cache=(a.CPP_SDK_job/'CPP-build/CMakeCache.txt').read_text()
        assert 'GGML_CUDA:BOOL=ON' in cache and 'GGML_CUDA_FORCE_CUBLAS:BOOL=OFF' in cache
        binaries={'ET':Path(et['ET_binary']),'CPP-default':a.CPP_SDK_job/'roofline_cpp_default'}
        assert digest(binaries['CPP-default'])==cp['binary_SHA256']
        et_prov=json.loads((a.ET_SDK_job/'ET-provenance.json').read_text())
        assert digest(binaries['ET'])==et_prov['binary'][str(binaries['ET'])]
        provenance=dict(CPP=cp,ET=et_prov,GGUF_SHA256=digest(a.gguf),arm_request_SHA256=digest(a.arms),artifact_arms=[],CPP_prefill_policy='stock default native CUDA routing; A16 is not forced')
        for arm in arms:
            wait_ready(Path(arm['export_job'])/'progress.json',lambda x:x.get('status')=='completed')
            artifact=Path(arm['artifact']);audit=json.loads((artifact/'artifact-audit.json').read_text())
            assert audit['GGUF_sha256']==provenance['GGUF_SHA256'] and audit['measured_context_before_step']==1024
            hashes={name:digest(artifact/name) for name in ['model.pte','aoti_cuda_blob.ptd']}
            assert all(hashes[name]==audit['artifacts'][name]['sha256'] for name in hashes)
            provenance['artifact_arms'].append(dict(**arm,hashes=hashes,artifact_audit_SHA256=digest(artifact/'artifact-audit.json')))
            arm['static_methods']=audit['static_methods']
        (a.job/'input-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        samples={x['name']:[] for x in arms};samples['CPP-default']=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        state('waiting-GPU-lock')
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);teardown()
            hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,power.limit,driver_version','--format=csv,noheader'],text=True).strip()
            provenance['hardware']=hardware;(a.job/'input-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            for round_id in range(1,5):
                order=list(samples) if round_id%2 else list(reversed(samples))
                for name in order:
                    state('measuring',round=round_id,framework=name,first_round_discarded=True)
                    output=a.job/f'{name}-round{round_id}.json'
                    if name=='CPP-default':command=[str(binaries[name]),str(a.gguf),str(output),str(a.repetitions)]
                    else:
                        arm=next(x for x in arms if x['name']==name);artifact=Path(arm['artifact'])
                        command=[str(binaries['ET']),str(artifact/'model.pte'),str(artifact/'aoti_cuda_blob.ptd'),str(output),str(a.repetitions)]
                    with (a.job/f'{name}-round{round_id}.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                    data=json.loads(output.read_text());assert data['completed'] and len(data['cases'])==7
                    assert data['history_population']=='only-current-mix; reset-between-cases'
                    assert all(c['same_prefix_greedy_repeats_equal'] and c['context_before_every_timed_step']==1024 and c['populated_private_histories']==c['decode_rows']+bool(c['prefill_rows']) for c in data['cases'])
                    if name!='CPP-default':
                        for c in data['cases']:
                            deltas=c['actual_method_call_deltas'];assert sum(deltas.values())==a.repetitions+3
                            widths=next(x for x in arms if x['name']==name)['static_methods']
                            fits=[m for m in widths if m>=c['batch_tokens']]
                            expected='forward_'+str(min(fits)) if fits else 'forward_others'
                            assert deltas[expected]==a.repetitions+3
                    samples[name].append(data);teardown()
        results=[]
        for name,rounds in samples.items():
            for i in range(7):
                cases=[v['cases'][i] for v in rounds];assert len({v['mix'] for v in cases})==1
                ms=[statistics.mean(c['latency_ms']) for c in cases];tps=[cases[0]['batch_tokens']*1000/t for t in ms]
                results.append(dict(framework=name,mix=cases[0]['mix'],batch_tokens=cases[0]['batch_tokens'],latency_ms_rounds=ms,processed_TPS_rounds=tps,latency_ms_mean=statistics.mean(ms[1:]),latency_ms_sample_SD=statistics.stdev(ms[1:]),processed_TPS_mean=statistics.mean(tps[1:]),processed_TPS_sample_SD=statistics.stdev(tps[1:]),first_round_discarded=True,greedy_tokens_per_round=[c['greedy_tokens'] for c in cases]))
        (a.job/'results.json').write_text(json.dumps(dict(rounds=4,discarded=[1],averaged=[2,3,4],results=results,comparison_scope='synchronized official API forward plus requested device greedy return',CPP_FORCE_CUBLAS=False,CPP_A16_not_forced=True,not_HTTP_output_throughput=True,full_task_F1_not_qualified_here=True,main_results_changed=False,hardware=hardware),indent=2)+'\n')
        state('completed-pending-independent-review',result_cells=len(results))
    except BaseException as error:state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
