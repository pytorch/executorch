"""Qualify runtime KV columns through the canonical Gemma HTTP API.

Wait for the separate model export and qualified worker. Use the same agent
client and GMRID inputs as the original comparison. Retain lifecycle traces,
four performance repeats and the agreed full/sampled correctness coverage.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['export-job','worker-job','artifact','et-source','tokenizer','prompts',
              'correctness-root','lock','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--server-python',required=True)
    p.add_argument('--benchmark-python',required=True)
    p.add_argument('--port',type=int,default=19121)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        tmp=a.output/'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_promoted=False,**extra),indent=2)+'\n');tmp.replace(a.output/'progress.json')
    def run(cmd,label,timeout):
        with (a.output/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(v) for v in cmd],stdout=log,stderr=subprocess.STDOUT)
            state('running-'+label,child_pid=child.pid)
            try:assert child.wait(timeout=timeout)==0,label
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=120)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
    try:
        state('waiting-model-and-worker')
        deadline=time.monotonic()+21600
        for job,accepted in [(a.worker_job,'CPU-HTTP-worker-ready-pending-model'),
                              (a.export_job,'completed')]:
            while True:
                f=job/'progress.json'
                if f.exists():
                    v=json.loads(f.read_text())
                    if v['status']==accepted:break
                    assert v['status']!='failed',v
                    os.kill(v['controller_pid'],0)
                assert time.monotonic()<deadline,'Wait deadline expired'
                time.sleep(15)
        assert sha(a.prompts)=='76d3ff55f4026a50679fd60becd07a9cad93549dc46d98a59bcd642c90d0f81e'
        assert sha(package/'benchmark_gemma.py')=='d2f9eba743b583b9b5968d2f8704cb347638a05fe12717797d21a8c938a00b05'
        worker=json.loads((a.worker_job/'build-provenance.json').read_text())
        assert sha(worker['binary'])==worker['binary_SHA256']
        audit=json.loads((a.artifact/'artifact-audit.json').read_text())
        assert audit['constant_metadata']['get_n_caches']==48
        assert audit['context']==131072 and audit['safe_ring_rows']==1535
        assert audit['selected_output_bound']==[64,262144]
        assert audit['static_methods']==[1,2,4,8,16,32,64]
        assert audit.get('runtime_physical_columns') is True or audit.get('runtime_columns') is True
        for name,item in audit['artifacts'].items():assert sha(a.artifact/name)==item['sha256']
        command=[a.server_python,'-B',str(package/'serve_gemma_runtime_columns.py'),
            '--worker-bin',worker['binary'],'--model-path',str(a.artifact/'model.pte'),
            '--data-path',str(a.artifact/'aoti_cuda_blob.ptd'),
            '--tokenizer-path',str(a.tokenizer),'--hf-tokenizer',str(a.tokenizer),
            '--max-context','131072','--initial-kv-capacity','512',
            '--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16',
            '--prefix-checkpoint-interval','512','--idle-kv-compaction-ms','250',
            '--idle-kv-spare-columns','1']
        wrapper=a.output/'serve-runtime-columns.sh'
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'test "${1:?backend}" = executorch\n'
            'entries=0\nif test "${SB_PREFIX_CACHE:-off}" = on; then entries=1; fi\n'
            'export PYTHONDONTWRITEBYTECODE=1\n'
            'export PYTHONPATH='+shlex.quote(str(a.et_source.parent))+'\n'
            'exec '+shlex.join(command)+' --prefix-cache-entries "$entries" --port "${SB_PORT:?port}"\n')
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        protocol=dict(context=131072,initial_KV=512,active_sessions=16,decode_ceiling=16,
            max_prefill_tokens=512,static_graphs=True,prefix_entries={'off':0,'on':1},
            checkpoint={'off':0,'on':512},idle_compaction_ms=250,idle_spare_columns=1,
            wrapper_SHA256=sha(wrapper),client_SHA256=sha(package/'benchmark_gemma.py'),
            prompts_SHA256=sha(a.prompts),worker_provenance=worker,model_promoted=False,
            performance_host_must_match_baseline=True)
        (a.output/'server-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        run([a.benchmark_python,'-B',package/'trace_gemma_runtime_http_memory.py',
             '--wrapper',wrapper,'--port',a.port,'--lock',a.lock,'--output',a.output/'lifecycle'],
            'real-HTTP-memory-lifecycle',7200)
        cmd=[a.benchmark_python,'-B',package/'run_matrix_24h.py','--wrapper',wrapper,
             '--model','gemma4-12b-it','--client',package/'benchmark_gemma.py',
             '--prompts',a.prompts,'--correctness-root',a.correctness_root,
             '--python',a.benchmark_python,'--lock',a.lock,'--port',a.port,
             '--variants','executorch','--cache-modes','off','on','--concurrencies','16','8','4','2','1']
        run(cmd+['--output',a.output/'server-performance','--phases','performance'],'server-performance',21600)
        run(cmd+['--output',a.output/'server-correctness','--phases','correctness'],'server-correctness',57600)
        state('completed-pending-independent-selection',real_HTTP=True,
              performance_memory_and_F1_collected=True)
    except BaseException as e:state('failed',error=str(e));raise


if __name__=='__main__':main()
