"""Profile fixed and dynamic physical KV after the paired HTTP qualification.

Keep every model, worker, HTTP source and benchmark client unchanged. Four
rounds of the first16 original agent prompts are allocation diagnostics only;
their latency and peaks never replace primary measurements. The existing GPU
lock covers each profiler's CUDA process teardown.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['fixed-job','dynamic-job','client','prompts','lock','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        f=a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            primary_results_unchanged=True,profile_is_allocation_diagnostic=True,**extra),indent=2)+'\n')
        f.replace(a.output/'progress.json')
    def wait(job):
        deadline=time.monotonic()+21600
        while True:
            f=job/'progress.json'
            if f.exists():
                value=json.loads(f.read_text())
                if value['status']=='completed-pending-independent-selection':return value
                assert value['status']!='failed',value
                os.kill(value['controller_pid'],0)
            assert time.monotonic()<deadline,'Observation deadline; preserve live qualification'
            time.sleep(30)
    try:
        state('waiting-existing-paired-HTTP-qualification')
        terminal={name:wait(job) for name,job in [('fixed',a.fixed_job),('dynamic',a.dynamic_job)]}
        assert sha(a.client)=='d2f9eba743b583b9b5968d2f8704cb347638a05fe12717797d21a8c938a00b05'
        assert sha(a.prompts)=='76d3ff55f4026a50679fd60becd07a9cad93549dc46d98a59bcd642c90d0f81e'
        protocols={name:json.loads((job/'server-protocol.json').read_text())
                   for name,job in [('fixed',a.fixed_job),('dynamic',a.dynamic_job)]}
        assert protocols['fixed']['worker_provenance']['binary_SHA256']==protocols['dynamic']['worker_provenance']['binary_SHA256']
        (a.output/'input-provenance.json').write_text(json.dumps(dict(terminal_qualification=terminal,
            server_protocols=protocols,source_client_SHA256=sha(a.client),source_prompts_SHA256=sha(a.prompts),
            selected_indices=list(range(16)),full_100_request_performance_not_repeated=True,
            Nsight_changes_timing=True),indent=2)+'\n')
        for name,job in [('fixed',a.fixed_job),('dynamic',a.dynamic_job)]:
            protocol=protocols[name];wrapper=job/'serve-runtime-columns.sh'
            assert protocol['runtime_physical_columns']==(name=='dynamic')
            assert protocol['initial_KV']==512 and protocol['context']==131072 and protocol['static_graphs']
            assert protocol['active_sessions']==16
            assert '--max-decode-sequences 16' in wrapper.read_text()
            assert sha(wrapper)==protocol['wrapper_SHA256']
            cutoff=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=40)
            command=[sys.executable,'-B',str(package/'profile_et_server.py'),
                '--wrapper',str(wrapper),'--wrapper-sha256',sha(wrapper),
                '--client',str(a.client),'--client-sha256',sha(a.client),
                '--prompts',str(a.prompts),'--prompts-sha256',sha(a.prompts),
                '--indices',*[str(i) for i in range(16)],'--concurrency','16',
                '--python',a.python,'--port',str(a.port),'--lock',str(a.lock),
                '--memory-only','--deadline-utc',cutoff.isoformat(),
                '--output',str(a.output/name)]
            with (a.output/(name+'-controller.log')).open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                state('profiling-'+name,child_pid=child.pid)
                try:assert child.wait(timeout=3000)==0,name
                finally:
                    if child.poll() is None:
                        child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=600)
                        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=60)
            result=json.loads((a.output/name/'progress.json').read_text())
            assert result['status']=='completed' and result['request_errors']==0
        state('completed-pending-independent-allocation-review',arms=['fixed','dynamic'],
              rounds=4,discarded=[1],averaged=[2,3,4],requests_per_arm=64,
              model_weights_and_native_sources_unchanged=True)
    except BaseException as e:state('failed',error=str(e));raise


if __name__=='__main__':main()
