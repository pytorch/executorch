"""Profile the existing HTTP/native routing on a saved MG mini, prefix OFF."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
import urllib.request


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    job, base = args.job, args.base
    assert job.is_dir() and not (job/'effective-config.json').exists()
    server = None
    rc = 1
    try:
        cfg = json.loads((base/'four-opt-MG-chat-main64-all-optimizations-20261028-v4/effective-config.json').read_text())
        artifact = base/'four-opt-MG-mini81-stable-layout128-qualification-20261028-v1/same-PTE'
        cfg.update(PTE=str(artifact/'model.pte'), PTD=str(artifact/'aoti_cuda_blob.ptd'),
            worker=str(base/'four-opt-MG-observable-full-native-worker-20261028-v5/worker'),
            model_id='muse-glimmer-mini-routing-diagnostic', prefix_checkpoint_interval=0)
        bindings = {name:dict(path=cfg[name], SHA256=digest(cfg[name]))
                    for name in ['PTE','PTD','worker']}
        (job/'effective-config.json').write_text(json.dumps(cfg,indent=2))
        (job/'bindings.json').write_text(json.dumps(dict(artifacts=bindings,mini_only=True,
            prefix_entries=0,active_sessions=64,decode_limit=64,prefill_chunk=4096,
            kernel_math_unchanged=True,only_existing_Runner_observation=True,
            real_HTTP_and_tokenizer=True,full_main_unchanged=True),indent=2))
        env=dict(os.environ,SB_PREFIX_CACHE='off',SB_PORT=str(args.port))
        command=[cfg['HTTP_python'],'-B','-u',str(base/'four-opt-MG-chat-main64-all-optimizations-20261028-v4/serve_four_opt_main.py'),str(job/'effective-config.json')]
        (job/'server-command.json').write_text(json.dumps(command,indent=2))
        url='http://127.0.0.1:'+str(args.port)
        with (base/'gpu-benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            with (job/'server.log').open('x') as log:
                server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            proc=Path('/proc')/str(server.pid)
            handle=dict(PID=server.pid,start_ticks=int((proc/'stat').read_text().split(') ',1)[1].split()[19]),
                command_SHA256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
            (job/'server-launch-receipt.json').write_text(json.dumps(handle,indent=2))
            deadline=time.monotonic()+120
            while True:
                assert server.poll() is None,'HTTP server exited before readiness'
                try:
                    with urllib.request.urlopen(url+'/health',timeout=2) as response:
                        if response.status==200:break
                except Exception:
                    assert time.monotonic()<deadline,'HTTP server readiness timeout'
                    time.sleep(.2)
            # Fixed public text: all rounds and concurrency levels use the same request.
            prompt='Synthetic throughput probe. Continue with neutral prose. '+('the ' * 1000)
            payload=dict(model=cfg['model_id'],messages=[dict(role='user',content=prompt)],
                max_tokens=64,temperature=0,stream=True,stream_options=dict(include_usage=True))
            body=json.dumps(payload).encode()
            (job/'request.json').write_bytes(body)
            def request(folder,owner,barrier):
                if barrier:barrier.wait()
                begin=time.monotonic_ns();first_byte=None;first_content=None;usage=None;finish=None;chunks=[]
                req=urllib.request.Request(url+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=300) as response:
                    for line in response:
                        if first_byte is None:first_byte=time.monotonic_ns()
                        chunks.append(line)
                        if not line.startswith(b'data: '):continue
                        data=line[6:].strip()
                        if data==b'[DONE]':continue
                        event=json.loads(data)
                        if event.get('usage'):usage=event['usage']
                        for choice in event.get('choices',[]):
                            if choice.get('delta',{}).get('content') and first_content is None:first_content=time.monotonic_ns()
                            if choice.get('finish_reason'):finish=choice['finish_reason']
                end=time.monotonic_ns();raw=b''.join(chunks)
                (folder/(str(owner)+'.sse')).write_bytes(raw)
                result=dict(owner=owner,begin_ns=begin,end_ns=end,first_byte_ns=first_byte,
                    first_content_ns=first_content,usage=usage,finish_reason=finish,
                    raw_SHA256=hashlib.sha256(raw).hexdigest())
                (folder/(str(owner)+'.json')).write_text(json.dumps(result,indent=2))
                assert usage is not None and usage['completion_tokens']==64,(owner,usage,finish)
                return result
            warm=job/'warmup';warm.mkdir();request(warm,0,None)
            rows=[]
            for round_id in range(1,5):
                for c in [1,2,4,8,16,32,64]:
                    folder=job/'cases'/str(round_id)/str(c);folder.mkdir(parents=True)
                    (job/'progress.json').write_text(json.dumps(dict(status='profiling-real-HTTP-mini',
                        round=round_id,c=c,complete=len(rows),prefix_entries=0,mini_only=True)))
                    barrier=threading.Barrier(c+1)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                        pending=[executor.submit(request,folder,owner,barrier) for owner in range(c)]
                        begin=time.monotonic_ns();barrier.wait();results=[f.result() for f in pending];end=time.monotonic_ns()
                    prompts={v['usage']['prompt_tokens'] for v in results};assert len(prompts)==1
                    record=dict(round=round_id,c=c,begin_ns=begin,end_ns=end,
                        throughput_tok_s=64*c/((end-begin)/1e9),prompt_tokens=next(iter(prompts)),
                        completion_tokens_each=64,requests=results)
                    (folder/'result.json').write_text(json.dumps(record,indent=2));rows.append(record)
                    (job/'records.json').write_text(json.dumps(rows,indent=2))
                    time.sleep(.3)
            assert len({row['prompt_tokens'] for row in rows})==1
            (job/'result.json').write_text(json.dumps(dict(records=rows,mini_only=True,
                real_HTTP_and_tokenizer=True,output_lengths_verified=True,full_main_unchanged=True,
                qualification=False,raw_independent_review_required=True),indent=2));rc=0
    except BaseException:
        (job/'failure.txt').write_text(traceback.format_exc())
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
        (job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_main_unchanged=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--port',type=int,default=20269)
    main(parser.parse_args())
