"""Trace an actual HTTP workload as runtime KV ownership grows and becomes idle.

NVML sampling includes the entire device, including allocator reservations.
Native worker logs separately record actual physical-column resize events.
This lifecycle diagnostic does not replace the standard throughput benchmark.
"""
import argparse
import concurrent.futures
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request


class NVML:
    class Memory(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
    def used(self):
        value=self.Memory();assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(value))==0
        return value.used


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['wrapper','lock','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    nvml=NVML();rows=[];results=[];stage='before';stop=threading.Event();started=time.monotonic()
    def sample():
        while not stop.is_set():
            rows.append(dict(seconds=time.monotonic()-started,stage=stage,GPU_used_bytes=nvml.used()))
            stop.wait(.05)
    def state(status,**extra):
        (a.output/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**extra),indent=2)+'\n')
    def request(barrier,index):
        body=dict(model='gemma4-12b-it',messages=[dict(role='user',content=
            'Repeat the word orange in a numbered list from 1 to 100. Do not explain. Request '+str(index))],
            stream=True,max_tokens=128,temperature=0,top_p=1,chat_template_kwargs=dict(enable_thinking=False))
        req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        barrier.wait();then=time.monotonic();events=[]
        with urllib.request.urlopen(req,timeout=600) as response:
            for line in response:
                text=line.decode().strip()
                if text.startswith('data: '):
                    if text=='data: [DONE]':break
                    events.append(json.loads(text[6:]))
        assert events and not any('error' in event for event in events),events
        text=''.join(v.get('choices',[{}])[0].get('delta',{}).get('content','') or '' for v in events)
        assert text.strip(),'Empty generated response'
        return dict(request=index,seconds=time.monotonic()-then,response=text,events=events)
    thread=threading.Thread(target=sample,daemon=True);thread.start();server=None
    try:
        with a.lock.open('a') as lock:
            state('waiting-GPU-lock');fcntl.flock(lock,fcntl.LOCK_EX)
            deadline=time.monotonic()+120
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<deadline,'Leave prior GPU work untouched';stop.wait(1)
            for prefix in ['off','on']:
                stage=prefix+'-loading';state('starting-server',prefix=prefix)
                env={**os.environ,'SB_PREFIX_CACHE':prefix,'SB_PORT':str(a.port)}
                log=(a.output/(prefix+'-server.log')).open('x')
                server=subprocess.Popen(['bash',str(a.wrapper),'executorch'],env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    deadline=time.monotonic()+2400
                    while True:
                        assert server.poll() is None,'Server exited; inspect log'
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/models',timeout=3) as r:
                                assert json.load(r)['data'];break
                        except Exception:
                            assert time.monotonic()<deadline;stop.wait(2)
                    stage=prefix+'-ready';stop.wait(2)
                    for step,c in enumerate([1,2,4,8,16,1]):
                        stage=prefix+'-step'+str(step)+'-c'+str(c)+'-active';state('running',prefix=prefix,concurrency=c)
                        barrier=threading.Barrier(c)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                            responses=list(pool.map(lambda i:request(barrier,i),range(c)))
                        results.append(dict(prefix=prefix,c=c,responses=responses))
                        stage=prefix+'-step'+str(step)+'-c'+str(c)+'-idle';stop.wait(3)
                finally:
                    stage=prefix+'-shutting-down'
                    if server.poll() is None:
                        os.killpg(server.pid,signal.SIGTERM)
                        try:server.wait(timeout=120)
                        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
                    log.close();server=None;stage=prefix+'-shutdown';stop.wait(2)
            state('completed-pending-independent-review',actual_HTTP=True)
    except BaseException as e:state('failed',error=str(e));raise
    finally:
        stop.set();thread.join(timeout=5)
        (a.output/'memory-samples.json').write_text(json.dumps(rows)+'\n')
        (a.output/'responses.json').write_text(json.dumps(results)+'\n')
        (a.output/'measurement.json').write_text(json.dumps(dict(sample_interval_seconds=.05,
            source='NVML total device memory; native logs report actual column widths',
            concurrency_sequence=[1,2,4,8,16,1],same_process_with_idle_between=True,
            prefix_modes=['off','on'],synthetic_lifecycle_only=True,
            standard_performance_and_F1_not_replaced=True),indent=2)+'\n')


if __name__=='__main__':main()
