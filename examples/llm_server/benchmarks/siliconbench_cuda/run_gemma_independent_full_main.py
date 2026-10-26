"""Prepare full MG/Gemma quality/performance runs after actual lifecycle gates.

Public source-only controller; model artifacts and results stay on the host.
Quality collection does not promote a candidate. Performance requires a full
qualification receipt whose raw evidence and current binaries are reverified.
"""
import argparse,datetime,json,os,shlex,subprocess,sys,threading,time,traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['serve','quality','performance'],required=True)
    parser.add_argument('--full-model-profile', choices=('gemma4-48layers','muse-glimmer-52layers'),
                        default='gemma4-48layers')
    for name in ('package','HTTP-job','native-job','memory-worker-proof',
                 'http-python-dependencies-job',
                 'tokenizer','server-python','public-inputs','lock','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('fixed-job','Gemma-source-preparation-job','native-lifecycle-job','mini-qualification'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--full-qualification-receipt',type=Path)
    parser.add_argument('--port',type=int,default=19910)
    a=parser.parse_args()
    assert 1024<=a.port<=65535
    sys.path.insert(0,str(a.package.resolve()))
    from independent_session_full_native_binding import sha,verify_runtime
    from independent_session_full_http_binding import verified_gemma_full_http, verified_mg_full_http
    gemma=a.full_model_profile=='gemma4-48layers'
    assert (a.fixed_job and a.Gemma_source_preparation_job) if gemma else (
        a.native_lifecycle_job and a.mini_qualification)
    a.mini_profile='gemma4-first6' if gemma else 'muse-glimmer-first4'
    a.full_native_job=a.native_job
    a.preparation_job=a.fixed_job/'full-pair' if gemma else a.native_lifecycle_job/'after-full-artifact'
    a.full_MG_native_lifecycle_job=a.native_lifecycle_job
    a.full_MG_mini_qualification=a.mini_qualification
    a.arm='independent';a.preserved_global_HTTP_failure_job=None
    verify_http=verified_gemma_full_http if gemma else verified_mg_full_http
    model_id='gemma4-12b-it' if gemma else 'muse-glimmer-30b'
    a.disable_native_memory_sampling=False
    if a.phase=='serve':
        binding=verify_http(a,a.package)
        prefix=os.environ['SB_PREFIX_CACHE'];assert prefix in ('off','on')
        port=int(os.environ['SB_PORT']);assert 1024<=port<=65535
        artifact=binding['artifacts']['independent']
        serve=a.package/'serve_gemma_runtime_columns.py' if gemma else binding['source']/'examples/models/muse-glimmer/serving/serve_batching.py'
        command=[str(a.server_python),'-B',str(serve),
            '--worker-bin',str(binding['worker']),'--model-path',str(artifact/'model.pte'),
            '--data-path',str(artifact/'aoti_cuda_blob.ptd'),
            '--tokenizer-path',str(a.tokenizer),'--hf-tokenizer',str(a.tokenizer),
            '--model-id',model_id,'--max-context','131072',
            '--max-sessions','16',
            '--max-decode-sequences','16','--max-inflight-requests','16',
            '--prefix-cache-entries',('1' if gemma else '4') if prefix=='on' else '0',
            '--prefix-checkpoint-interval','512' if gemma and prefix=='on' else '0',
            '--tool-parser','none','--host','127.0.0.1','--port',str(port)]
        if binding['explicit_initial']:command+=['--initial-kv-capacity','512']
        command+=['--raw-tool-text'] if gemma else ['--backend','cuda']
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
            ET_CUDA_BENCH_MEMORY_TRACE='1',
            PYTHONPATH=str(binding['dependency_path'])+os.pathsep+str(binding['source'].parent))
        print('FULL_MAIN_SERVER_BINDING:'+json.dumps(dict(argv=command,
            full_profile=a.full_model_profile,model_PTE_SHA256=sha(artifact/'model.pte'),
            native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),
            memory_worker_SHA256=sha(binding['worker']),prefix=prefix,
            static_forward_graphs_enabled=True,legacy_idle_policy_unchanged=True,
            server_binding_monotonic_ns=time.monotonic_ns(),
            declared_owner_limit=21,active_decode_limit=16,initial_KV=512,logical_context=131072)),flush=True)
        os.execvpe(command[0],command,env)
    assert not a.output.exists();a.output.mkdir(parents=True)
    def state(status,**extra):
        f=a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            phase=a.phase,full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        f.replace(a.output/'progress.json')
    try:
        deadline=time.monotonic()+12*3600
        while True:
            d=json.loads((a.HTTP_job/'progress.json').read_text())
            if d['status']=='full-HTTP-memory-collected-pending-F1-and-main':break
            assert d['status']!='failed',d
            proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
            assert proc.is_file() and any(name in proc.read_bytes() for name in (
                b'run_gemma_full_http_after_lifecycle.py', b'run_full_MG_HTTP_after_native.py'))
            assert time.monotonic()<deadline,'Preserve verified-live work; observation deadline reached'
            state('waiting-confirmed-live-full-HTTP',PID=d['controller_pid'],HTTP_status=d['status'])
            time.sleep(20)
        verify_runtime(a.native_job,a.package)
        parent=json.loads((a.HTTP_job/'result.json').read_text())
        assert parent['full_profile']==a.full_model_profile and parent['full_HTTP_memory_pass']
        assert sha(a.HTTP_job/'HTTP-independent-review.json')==parent['HTTP_review_SHA256']
        if parent['positive_performance_signals'] and a.phase=='performance' and gemma:
            state('full-performance-gate-not-passed',positive_performance_signals=parent['positive_performance_signals'])
            return
        binding=verify_http(a,a.package)
        artifact=binding['artifacts']['independent']
        input_progress=json.loads((a.public_inputs/'progress.json').read_text())
        assert input_progress['status']=='pinned-public-main-inputs-ready'
        benchmark_python=a.public_inputs/'.venv/bin/python';assert benchmark_python.is_file()
        dataset=json.loads((a.public_inputs/'GMRID-input-audit.json').read_text())
        assert dataset['evaluated_items']==1146
        # Input preparation stores exact downloaded bytes and absolute paths.
        prompt_audit=json.loads((a.public_inputs/'SiliconBench-input-audit.json').read_text())
        saved_paths=prompt_audit['splits']
        prompt_paths={}
        for split in ('chat','agent'):
            record=saved_paths[split]
            path=a.public_inputs/(split+'_benchmark_prompts.json')
            assert path.resolve().is_relative_to(a.public_inputs.resolve())
            assert sha(path)==record['sha256']
            prompt_paths[split]=path
        artifact_SHA={name:sha(artifact/name) for name in ('model.pte','aoti_cuda_blob.ptd')}
        if a.phase=='performance':
            assert a.full_qualification_receipt
            qualification=json.loads(a.full_qualification_receipt.read_text())
            for key in ('full_model_correctness_pass','full_task_F1_independently_reviewed',
                        'full_fixed_token_performance_pass','full_storage_rewind_graph_pass','full_HTTP_memory_pass'):
                assert qualification[key],key
            assert qualification['full_profile']==a.full_model_profile
            assert qualification['artifact_SHA256']==artifact_SHA
            assert qualification['native_runtime_gate_SHA256']==sha(a.native_job/'native-gate.json')
            assert not qualification['unresolved_correctness_or_performance_signals']
            for name,expected in qualification['evidence_files_SHA256'].items():
                assert sha(name)==expected,name
            if not gemma:
                from write_MG_full_independent_KV_qualification import verify_MG_full_qualification
                verify_MG_full_qualification(a.full_qualification_receipt,a,binding)
        # This generated wrapper remains in the owned experiment output only.
        common=[]
        common=['--full-model-profile',a.full_model_profile]
        for name in ('package','HTTP-job','native-job','memory-worker-proof',
                     'http-python-dependencies-job','tokenizer','server-python','public-inputs','lock','output',
                     *(('fixed-job','Gemma-source-preparation-job') if gemma else ('native-lifecycle-job','mini-qualification'))):
            common.extend(['--'+name,str(getattr(a,name.replace('-','_')))])
        serve=[str(a.server_python),'-B',str(Path(__file__).resolve()),'--phase','serve',*common]
        wrapper=a.output/'serve-qualified-full.sh'
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'test "${1:?backend}" = executorch\nexec '+shlex.join(serve)+'\n')
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        source_receipt=dict(full_profile=a.full_model_profile,artifact_SHA256=artifact_SHA,
            native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),wrapper_SHA256=sha(wrapper),
            native_allocator_sampler_enabled=True,sampler_interval_ms=250,
            HTTP_parent_result_SHA256=sha(a.HTTP_job/'result.json'),
            original_full_performance_signals=parent['positive_performance_signals'],
            unresolved_full_performance_signals=([] if a.phase=='performance' else parent['positive_performance_signals']),
            full_qualification_receipt_SHA256=sha(a.full_qualification_receipt) if a.phase=='performance' else None,
            full_model_qualified=False,main_candidate_accepted=False)
        (a.output/'provenance.json').write_text(json.dumps(source_receipt,indent=2)+'\n')
        completed=[]
        splits=('agent',) if a.phase=='quality' else ('chat','agent')
        for split in splits:
            for prefix in ('off','on'):
                for c in (16,1,8,4,2):
                    mode='correctness' if a.phase=='quality' else 'performance'
                    case=a.output/(mode+'-'+split+'-prefix-'+prefix+'-c'+str(c))
                    argv=[str(benchmark_python),'-B',str(a.package/'run_case_24h.py'),
                        '--wrapper',str(wrapper),'--backend','executorch','--model',model_id,
                        '--variant','independent-session-KV','--prefix-cache',prefix,'--client',str(a.package/('benchmark_gemma.py' if gemma else 'benchmark_mg.py')),
                        '--prompts',str(prompt_paths[split]),'--correctness-root',str(a.public_inputs/'correctness'),
                        '--python',str(benchmark_python),'--concurrency',str(c),'--port',str(a.port),
                        '--output',str(case),'--lock',str(a.lock),'--mode',mode,'--split',split]
                    if a.phase=='quality':
                        argv+=['--quality-scope','full' if c in (1,16) else 'smoke','--correctness-rounds','1']
                        if not gemma:
                            argv+=['--correctness-max-tokens','1024','--correctness-chat-template-kwargs',
                                   json.dumps(dict(enable_thinking=False,current_date='2026-10-21',reasoning_strength='low'))]
                    (a.output/(case.name+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
                    state('collecting-'+mode,split=split,prefix=prefix,concurrency=c,case=str(case),completed=completed)
                    # Observe the unchanged driver's atomic progress records.
                    # These timestamps align native sampler counters with each
                    # request round without changing benchmark/server code.
                    stop=threading.Event();errors=[]
                    def observe():
                        last=None;stopping=False
                        try:
                            with (a.output/(case.name+'-observed-phases.jsonl')).open('x') as out:
                                while True:
                                    progress_path=case/'progress.json'
                                    if progress_path.is_file():
                                        current=json.loads(progress_path.read_text())
                                        identity=tuple(current.get(k) for k in
                                            ('status','round','shots','server_pid','controller_pid'))
                                        if identity!=last:
                                            out.write(json.dumps(dict(observed_monotonic_ns=time.monotonic_ns(),
                                                observer_interval_ms=50,driver_progress=current))+'\n')
                                            out.flush();last=identity
                                    if stopping:break
                                    stopping=stop.wait(.05)
                        except Exception as error:errors.append(repr(error))
                    observer=threading.Thread(target=observe,daemon=True);observer.start()
                    try:
                        with (a.output/(case.name+'.log')).open('x') as log:
                            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                            try:
                                assert child.wait(timeout=43200)==0,'Case failed; inspect preserved driver logs'
                            finally:
                                if child.poll() is None:
                                    child.terminate()
                                    try:child.wait(timeout=120)
                                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    finally:
                        stop.set();observer.join(timeout=10)
                        assert not observer.is_alive() and not errors,errors
                    d=json.loads((case/'progress.json').read_text());assert d['status']=='completed',d
                    settings=json.loads((case/'settings.json').read_text())
                    assert settings['concurrency']==c and settings['split']==split
                    if a.phase=='quality':
                        assert settings['correctness_items']==(1146 if c in (1,16) else 64)
                        scores=json.loads((case/'correctness-scores.json').read_text())
                        assert len(scores)==2
                        assert {s['shots'] for s in scores}=={0,5}
                        assert all(s['n_scored']==settings['correctness_items'] and s['request_errors']==0 and
                            s['n_missing_responses']==s['n_extra_responses']==0 for s in scores)
                    else:
                        metrics=json.loads((case/'performance-aggregate.json').read_text())
                        assert metrics['successful']['mean']==100 and metrics['failed']['mean']==0
                    completed.append(case.name)
        expected=10 if a.phase=='quality' else 20;assert len(completed)==expected
        (a.output/'result.json').write_text(json.dumps(dict(source_receipt,completed=completed,
            raw_quality_acceptance_pending=a.phase=='quality',raw_main_acceptance_pending=True,
            full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
        state('collected-pending-independent-raw-review',completed=completed)
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
