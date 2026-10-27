"""Move an identical qualified Gemma runtime to its previous comparison GPU.

Keep origin qualification and performance results intact. Reuse only byte
identical weights, validate native CUDA and HTTP portability, and collect fresh
performance on the destination. Do not treat origin timings as target timings.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import xml.etree.ElementTree as ET


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()


def safe_relative(name):
    path=Path(name)
    assert not path.is_absolute() and '..' not in path.parts and path.parts
    assert path.parts[0] in ['artifact','runtime','origin','tokenizer','inputs']
    return path


def receive(a):
    manifest=json.loads(a.manifest.read_text())
    assert manifest['kind']=='exact-qualified-Gemma-user-migration'
    assert {'runtime/libaoti_cuda_shims.so','runtime/libextension_cuda.so'} <= set(manifest['files'])
    assert manifest['origin_full_qualification_verified_on_original_GPU']
    assert manifest['weights_not_transferred'] and manifest['source_origin_unmodified']
    assert not a.destination.exists()
    assert a.reuse_weights.is_file() and a.reuse_weights.stat().st_size==manifest['weights_bytes']
    assert sha(a.reuse_weights)==manifest['same_local_weights_required_SHA256']
    partial=a.destination.with_name(a.destination.name+'.partial');partial.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(partial).free>sum(r['bytes'] for r in manifest['files'].values())+1024**3
    seen=set()
    with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as archive:
        for member in archive:
            name=member.name;relative=safe_relative(name)
            assert member.isfile() and name in manifest['files'] and name not in seen
            record=manifest['files'][name];assert member.size==record['bytes']
            target=partial/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out,archive.extractfile(member) as incoming:
                shutil.copyfileobj(incoming,out,16*1024*1024)
            assert sha(target)==record['SHA256']
            target.chmod(record['mode']);seen.add(name)
    assert seen==set(manifest['files'])
    os.link(a.reuse_weights,partial/'artifact/aoti_cuda_blob.ptd')
    (partial/'migration-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (partial/'transfer-provenance.json').write_text(json.dumps(dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),every_received_file_SHA256_verified=True,
        exact_local_weights_reused=str(a.reuse_weights),source_and_existing_models_modified=False,
        source_manifest_SHA256=sha(a.manifest)),indent=2)+'\n')
    partial.rename(a.destination)
    print('TASK_JSON:'+json.dumps(dict(received=True,files=len(seen),weights_not_transferred=True)))


def binding(a):
    manifest=json.loads((a.destination/'migration-manifest.json').read_text())
    assert manifest['kind']=='exact-qualified-Gemma-user-migration'
    assert {'runtime/libaoti_cuda_shims.so','runtime/libextension_cuda.so'} <= set(manifest['files'])
    for name,record in manifest['files'].items():
        path=a.destination/safe_relative(name)
        assert path.stat().st_size==record['bytes'] and sha(path)==record['SHA256'],name
    assert sha(a.destination/'artifact/aoti_cuda_blob.ptd')==manifest['same_local_weights_required_SHA256']
    q=json.loads((a.destination/'origin/full-qualification.json').read_text())
    assert sha(a.destination/'origin/full-qualification.json')==manifest['origin_full_qualification_SHA256']
    for key in ['full_model_correctness_pass','full_task_F1_independently_reviewed',
                'full_storage_rewind_graph_pass','full_HTTP_memory_pass']:assert q[key],key
    assert q['full_fixed_token_performance_pass'] or q.get('full_fixed_token_performance_accepted_by_user')
    assert not q['unresolved_correctness_or_performance_signals']
    assert q['artifact_SHA256']['model.pte']==sha(a.destination/'artifact/model.pte')
    assert q['artifact_SHA256']['aoti_cuda_blob.ptd']==manifest['same_local_weights_required_SHA256']
    quality=json.loads((a.destination/'origin/full-task-raw-review.json').read_text())
    assert quality['complete'] and quality['actual_response_count']==9936 and quality['reviewed_shots']==20
    assert quality['full_task_F1_independently_reviewed'] and quality['full_task_raw_quality_pass']
    assert quality['artifact_SHA256']==q['artifact_SHA256']
    for row in quality['shots']:
        assert row['errors']==row['parse_failures']==row['final_content_scoring_disagreements']==0
        assert row['finish_reasons']=={'stop':row['count']}
    proof=json.loads((a.destination/'origin/memory-worker-proof.json').read_text())
    worker=a.destination/'runtime/gemma_runtime_columns_worker'
    assert sha(worker)==proof['instrumented_worker_SHA256']
    for original,expected in proof['unchanged_SDK_libraries'].items():
        if Path(original).suffix=='.so':assert sha(a.destination/'runtime'/Path(original).name)==expected
    # Export-only Python is not imported when executing the preserved PTE.
    HTTP={n:v for n,v in manifest['HTTP_source_SHA256'].items() if Path(n).name!='export_batching.py'}
    for name,expected in HTTP.items():assert sha(a.source/name)==expected,name
    package=Path(__file__).resolve().parent
    assert sha(package/'serve_gemma_runtime_columns.py')==manifest['serve_script_SHA256']
    assert sha(package/'benchmark_gemma.py')==manifest['benchmark_client_SHA256']
    query=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,power.limit,driver_version',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    assert query.split(',')[0].strip()==a.expected_GPU_UUID and 'RTX 5090' in query
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
             LD_LIBRARY_PATH=str(a.destination/'runtime')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
             PYTHONPATH=str(a.source.parent)+os.pathsep+os.environ.get('PYTHONPATH',''),
             ET_CUDA_BENCH_MEMORY_TRACE='1')
    linked=subprocess.run(['ldd',str(worker)],env=env,capture_output=True,text=True,check=True).stdout
    assert 'not found' not in linked,linked
    for name in ['libaoti_cuda_shims.so','libextension_cuda.so']:
        assert str(a.destination/'runtime'/name) in linked,name
    return manifest,q,env,query


def serve(a):
    manifest,q,env,query=binding(a)
    prefix=os.environ['SB_PREFIX_CACHE'];assert prefix in ['on','off']
    port=int(os.environ['SB_PORT']);assert 1024<=port<=65535
    package=Path(__file__).resolve().parent
    command=[sys.executable,'-B',str(package/'serve_gemma_runtime_columns.py'),
             '--worker-bin',str(a.destination/'runtime/gemma_runtime_columns_worker'),
             '--model-path',str(a.destination/'artifact/model.pte'),
             '--data-path',str(a.destination/'artifact/aoti_cuda_blob.ptd'),
             '--tokenizer-path',str(a.destination/'tokenizer'),'--hf-tokenizer',str(a.destination/'tokenizer'),
             '--model-id','gemma4-12b-it','--max-context','131072','--initial-kv-capacity','512',
             '--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16',
             '--prefix-cache-entries','1' if prefix=='on' else '0',
             '--prefix-checkpoint-interval','512' if prefix=='on' else '0',
             '--tool-parser','none','--raw-tool-text','--host','127.0.0.1','--port',str(port)]
    print('FULL_MAIN_SERVER_BINDING:'+json.dumps(dict(argv=command,
        full_profile='gemma4-48layers',model_PTE_SHA256=q['artifact_SHA256']['model.pte'],
        native_runtime_gate_SHA256=sha(a.destination/'origin/native-gate.json'),
        memory_worker_SHA256=sha(a.destination/'runtime/gemma_runtime_columns_worker'),prefix=prefix,
        static_forward_graphs_enabled=True,legacy_idle_policy_unchanged=True,
        server_binding_monotonic_ns=time.monotonic_ns(),declared_owner_limit=21,
        active_decode_limit=16,initial_KV=512,logical_context=131072,
        migration_same_artifact_runtime=True,target_GPU_query=query,
        origin_qualification_GPU_UUID=manifest['origin_GPU_UUID'])),flush=True)
    os.execvpe(command[0],command,env)


def collect(a):
    manifest,q,env,query=binding(a)
    a.output.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parent
    def state(status,**values):
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            controller_pid=os.getpid(),origin_full_F1_reused=True,main_candidate_accepted=False,
            target_GPU=query,**values),indent=2)+'\n');pending.replace(a.output/'progress.json')
    def interrupted(signum,frame):raise InterruptedError('User interrupted migration collection')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        state('checking-native-CUDA-portability')
        test=a.destination/'runtime/test_cuda_kv_cache'
        xml=a.output/'target-native-tests.xml'
        with (a.output/'native-tests.log').open('x') as log:
            subprocess.run([str(test),'--gtest_output=xml:'+str(xml)],env=env,stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=600)
        tests=list(ET.parse(xml).getroot().iter('testcase'))
        assert len(tests)==74 and not any(any(t.find(k) is not None for k in ['failure','error','skipped']) for t in tests)
        wrapper=a.output/'serve-qualified-full.sh'
        common=['--destination',str(a.destination),'--source',str(a.source),'--expected-GPU-UUID',a.expected_GPU_UUID]
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntest "${1:?}" = executorch\nexec '+
                           shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--phase','serve',*common])+'\n')
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        provenance=dict(origin_qualification_SHA256=manifest['origin_full_qualification_SHA256'],
            artifact_SHA256=q['artifact_SHA256'],worker_SHA256=sha(a.destination/'runtime/gemma_runtime_columns_worker'),
            target_GPU=query,source_and_artifact_identical=True,target_native_tests=74,
            target_native_tests_XML_SHA256=sha(xml),origin_timings_not_claimed_as_target_timings=True,
            previous_target_old_results_reused=True,four_rounds_discard_first=True)
        (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        completed=[]
        for mode in ['correctness','performance']:
            for split in (['agent'] if mode=='correctness' else ['chat','agent']):
                for prefix in ['off','on']:
                    for concurrency in ([16,1] if mode=='correctness' else [16,1,8,4,2]):
                        case=a.output/(mode+'-'+split+'-prefix-'+prefix+'-c'+str(concurrency))
                        argv=[sys.executable,'-B',str(package/'run_case_24h.py'),
                            '--wrapper',str(wrapper),'--backend','executorch','--model','gemma4-12b-it',
                            '--variant','independent-session-KV-migrated-identical-runtime','--prefix-cache',prefix,
                            '--client',str(package/'benchmark_gemma.py'),'--prompts',str(a.destination/'inputs'/(split+'_benchmark_prompts.json')),
                            '--correctness-root',str(a.destination/'inputs/correctness'),'--python',sys.executable,
                            '--concurrency',str(concurrency),'--port',str(a.port),'--output',str(case),
                            '--lock',str(a.lock),'--mode',mode,'--split',split]
                        if mode=='correctness':argv+=['--quality-scope','smoke','--correctness-rounds','1']
                        state('collecting-'+mode,case=str(case),completed=completed)
                        (a.output/(case.name+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
                        stop=threading.Event();errors=[]
                        def observe():
                            last=None
                            try:
                                with (a.output/(case.name+'-observed-phases.jsonl')).open('x') as f:
                                    while True:
                                        p=case/'progress.json'
                                        if p.is_file():
                                            d=json.loads(p.read_text());identity=tuple(d.get(k) for k in ['status','round','shots','server_pid','controller_pid'])
                                            if identity!=last:
                                                f.write(json.dumps(dict(observed_monotonic_ns=time.monotonic_ns(),observer_interval_ms=50,driver_progress=d))+'\n');f.flush();last=identity
                                        if stop.wait(.05):break
                            except Exception as error:errors.append(repr(error))
                        observer=threading.Thread(target=observe,daemon=True);observer.start()
                        try:
                            with (a.output/(case.name+'.log')).open('x') as log:
                                child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                                try:assert child.wait(timeout=43200)==0,'Case failed; preserve logs'
                                finally:
                                    if child.poll() is None:
                                        child.terminate()
                                        try:child.wait(timeout=120)
                                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        finally:stop.set();observer.join(timeout=10);assert not observer.is_alive() and not errors,errors
                        assert json.loads((case/'progress.json').read_text())['status']=='completed'
                        if mode=='correctness':
                            scores=json.loads((case/'correctness-scores.json').read_text());assert len(scores)==2
                            assert {s['shots'] for s in scores}=={0,5}
                            assert all(s['n_scored']==64 and s['request_errors']==0 and s['n_missing_responses']==s['n_extra_responses']==0 for s in scores)
                        else:
                            result=json.loads((case/'performance-aggregate.json').read_text())
                            assert result['successful']['mean']==100 and result['failed']['mean']==0
                        completed.append(case.name)
        state('migration-native-HTTP-smoke-and-twenty-main-cases-collected',completed=completed)
        (a.output/'result.json').write_text(json.dumps(dict(main_cases_collected=20,portability_smoke_cases=4,
            original_full_F1_preserved=True,new_full_F1_not_claimed=True,previous_same_GPU_old_results_reused=True,
            new_main_independent_raw_review_pending=True,provenance=provenance),indent=2)+'\n')
    except BaseException as error:
        state('failed',error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['receive','serve','collect'],required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--reuse-weights',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--expected-GPU-UUID')
    p.add_argument('--output',type=Path);p.add_argument('--lock',type=Path);p.add_argument('--port',type=int,default=19962)
    a=p.parse_args()
    if a.phase=='receive':assert a.manifest and a.reuse_weights;receive(a)
    else:
        assert a.source and a.expected_GPU_UUID
        if a.phase=='serve':serve(a)
        else:assert a.output and a.lock;collect(a)


if __name__=='__main__':main()
