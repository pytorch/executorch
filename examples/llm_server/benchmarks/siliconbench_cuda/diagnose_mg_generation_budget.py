"""Isolated MG generation controls; preserve the main GMRID protocol/results.

Same official prompts and stock llama-server. Compare the official template
reasoning_strength argument and output budget on three deterministic items.
This is a generation diagnostic, never a replacement full F1 or performance
cell. A fresh server and the existing GPU lock isolate it from other work.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper',type=Path,required=True)
    p.add_argument('--correctness-root',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=19108)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.mkdir();server=None;lock=a.lock.open('a')
    def state(status,**kw):
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),main_protocol_unchanged=True,**kw),indent=2)+'\n')
        pending.replace(a.output/'progress.json')
    def interrupted(signum,frame):raise InterruptedError('Clean up only this owned diagnostic server')
    signal.signal(signal.SIGTERM,interrupted)
    def get(path):
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}'+path,timeout=5) as response:return json.load(response)
    try:
        state('waiting-GPU-lock');fcntl.flock(lock,fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        inputs={}
        for shots in [0,5]:
            path=a.correctness_root/f'prompts/{shots}shot/prompts.jsonl'
            inputs[shots]=[json.loads(line) for line in path.open()]
            assert len(inputs[shots])==1146 and all(r['max_tokens']==256 for r in inputs[shots])
        labels=[json.loads(line) for line in (a.correctness_root/'prompts/0shot/labels.jsonl').open()]
        grouped={}
        for r in labels:grouped.setdefault(r['gold_category'],[]).append(str(r['id']))
        ids=[]
        for category in sorted(grouped)[:3]:
            ids.append(min(grouped[category],key=lambda i:hashlib.sha256(('MG-generation-controls-v1:'+i).encode()).hexdigest()))
        assert len(set(ids))==3
        provenance=dict(selected_ids=ids,source_prompts_sha256={str(s):hashlib.sha256((a.correctness_root/f'prompts/{s}shot/prompts.jsonl').read_bytes()).hexdigest() for s in inputs},wrapper_sha256=hashlib.sha256(a.wrapper.read_bytes()).hexdigest(),controls=[dict(reasoning_strength=s,max_tokens=b) for s,b in [('high',256),('high',1024),('low',256),('low',1024)]],temperature=0,top_p=1,concurrency=1,prefix_cache=False,no_source_patches=True,no_partial_F1_claim=True,no_performance_claim=True,main_correctness_budget=256)
        (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        env=dict(os.environ,SB_PORT=str(a.port),SB_PREFIX_CACHE='off',SB_CPP_BUILD='build-stock-native-evaluation',APPLEBENCH_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        with (a.output/'server.log').open('x') as log:
            server=subprocess.Popen(['bash',str(a.wrapper),'llamacpp'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state('starting-server',server_pid=server.pid)
        end=time.monotonic()+2400
        while True:
            assert server.poll() is None,'Owned server exited; inspect retained log'
            try:
                models=get('/v1/models')['data']
                if models:model=models[0]['id'];break
            except Exception:pass
            assert time.monotonic()<end,'Owned server did not become ready'
            time.sleep(5)
        rows=[]
        with (a.output/'responses.jsonl').open('x') as out:
            for strength,budget in [('high',256),('high',1024),('low',256),('low',1024)]:
                for shots in [0,5]:
                    for record in inputs[shots]:
                        if str(record['id']) not in ids:continue
                        payload=dict(model=model,messages=record['messages'],max_tokens=budget,temperature=0,top_p=1,stream=False,chat_template_kwargs=dict(enable_thinking=False,reasoning_strength=strength))
                        request=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                        state('running-generation-controls',completed=len(rows),total=24,reasoning_strength=strength,max_tokens=budget,shots=shots)
                        start=time.monotonic()
                        with urllib.request.urlopen(request,timeout=300) as response:result=json.load(response)
                        choice=result['choices'][0];message=choice['message'];content=message.get('content') or ''
                        final=content.split('to=user<|message|>',1)[1] if 'to=user<|message|>' in content else ''
                        parsed=None
                        if final:
                            decoder=json.JSONDecoder()
                            for i,ch in enumerate(final):
                                if ch!='{':continue
                                try:
                                    obj,_=decoder.raw_decode(final[i:])
                                    if isinstance(obj,dict) and 'category' in obj:parsed=obj['category'];break
                                except ValueError:pass
                        row=dict(id=str(record['id']),shots=shots,reasoning_strength=strength,max_tokens=budget,content=content,reasoning=message.get('reasoning_content') or message.get('reasoning') or '',usage=result.get('usage'),finish_reason=choice.get('finish_reason'),latency_s=time.monotonic()-start,has_final_channel=bool(final),final_JSON_category=parsed)
                        rows.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
        assert len(rows)==24
        groups=[]
        for strength,budget in [('high',256),('high',1024),('low',256),('low',1024)]:
            matched=[r for r in rows if (r['reasoning_strength'],r['max_tokens'])==(strength,budget)]
            groups.append(dict(reasoning_strength=strength,max_tokens=budget,requests=len(matched),finish_reasons=dict(collections.Counter(r['finish_reason'] for r in matched)),final_channel_count=sum(r['has_final_channel'] for r in matched),final_JSON_count=sum(r['final_JSON_category'] is not None for r in matched)))
        (a.output/'results.json').write_text(json.dumps(dict(provenance=provenance,groups=groups,interpretation='Small generation control only. Main 256-token official-scored F1 preserved; final-channel validity reported separately. No inference-kernel failure inferred.'),indent=2)+'\n')
        state('completed',requests=24)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:]);raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        if server is not None:
            until=time.monotonic()+60
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<until,'Wait for owned server CUDA teardown before releasing lock';time.sleep(1)
        lock.close()


if __name__=='__main__':main()
