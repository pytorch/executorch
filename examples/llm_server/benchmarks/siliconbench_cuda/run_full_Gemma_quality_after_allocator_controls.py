"""Collect same-PTE full Gemma ABBA and full task F1 after real allocator gates.

Retain numerical/performance signals and raw evidence. This workflow does not
qualify the full model or collect main performance configurations.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime,verify_qualification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-controls-job','native-job','before-fixed-job','mini-qualification',
                 'memory-worker-proof','Gemma-source-preparation-job','http-python-dependencies-job',
                 'tokenizer','public-inputs','lock','nsys','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    a=parser.parse_args();assert 1024<=a.port<=65535
    assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        p=a.output/'progress.pending.json'
        p.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        p.replace(a.output/'progress.json')
    def run(name,script,tail):
        argv=[sys.executable,'-B','-u',str(package/script),*map(str,tail)]
        (a.output/(name+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run(argv,check=True,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,timeout=24*3600)
    try:
        deadline=time.monotonic()+12*3600
        parent=a.full_controls_job
        while True:
            d=json.loads((parent/'progress.json').read_text())
            if d['status']=='full48-allocator-controls-collected-pending-F1-and-main':break
            assert d['status']!='failed',d
            handle=Path('/proc')/str(d['controller_pid'])/'cmdline'
            argv=handle.read_bytes().split(bytes([0])) if handle.is_file() else []
            if not argv and json.loads((parent/'progress.json').read_text())['status']=='full48-allocator-controls-collected-pending-F1-and-main':break
            assert any(Path(v.decode()).name=='run_gemma_full_after_allocator_mini.py' for v in argv if v)
            assert b'--output' in argv and argv[argv.index(b'--output')+1]==os.fsencode(parent)
            assert time.monotonic()<deadline,'Preserve confirmed-live full controls after observation timeout'
            state('waiting-confirmed-live-full-Gemma-controls',parent_PID=d['controller_pid'],parent_status=d['status'])
            time.sleep(20)
        native=verify_runtime(a.native_job,package)
        verify_qualification(a.mini_qualification,'gemma4-first6',
            '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a',a.native_job)
        parent_path=parent/'result.json';parent_hash=sha(parent_path)
        doc=json.loads(parent_path.read_text())
        assert doc['full_profile']=='gemma4-48layers'
        assert doc['native_runtime_gate_SHA256']==sha(a.native_job/'native-gate.json')
        for path,expected in doc['completed_result_SHA256'].items():assert sha(path)==expected
        fixed=parent/'full-fixed';HTTP=parent/'full-HTTP'
        lifecycle=json.loads((parent/'full-lifecycle/result.json').read_text())
        assert lifecycle['full_storage_rewind_graph_pass']
        assert json.loads((HTTP/'result.json').read_text())['full_HTTP_memory_pass']
        before=a.before_fixed_job
        assert json.loads((before/'progress.json').read_text())['status']=='full-fixed-token-collected-pending-other-full-gates'
        state('collecting-full-Gemma-same-PTE-native-ABBA')
        matched=a.output/'matched-control'
        run('matched-control','run_interleaved_native_clock_control.py',[
            '--before-preparation',before/'full-pair','--after-preparation',fixed/'full-pair',
            '--before-rewind-probe',before/'rewind-probe','--after-rewind-probe',fixed/'rewind-probe',
            '--mini-profile','gemma4-first6','--full-model-profile','gemma4-48layers',
            '--arm','independent','--lock',a.lock,'--nsys',a.nsys,'--output',matched])
        control=json.loads((matched/'result.json').read_text())
        assert control['full_model_profile']=='gemma4-48layers' and control['collected_arms']==['independent']
        assert control['four_rounds_discard_first'] and control['ABBA_native_order'] and control['initial_KV_capacity']==512
        review=json.loads((matched/'same-PTE-native-independent-review.json').read_text())
        assert review['full_model_profile']=='gemma4-48layers' and review['native_change_preserves_observed_model_numerics']
        assert review['full_saved_FP32_tensors_equal']==9 and review['selected_argmax_values_equal']>0
        state('collecting-fresh-full-Gemma-task-quality',matched_performance_signals=control['positive_performance_signals'])
        quality=a.output/'full-quality'
        run('full-quality','run_gemma_independent_full_main.py',[
            '--phase','quality','--full-model-profile','gemma4-48layers','--package',package,
            '--HTTP-job',HTTP,'--fixed-job',fixed,'--native-job',a.native_job,
            '--Gemma-source-preparation-job',a.Gemma_source_preparation_job,
            '--memory-worker-proof',a.memory_worker_proof,
            '--http-python-dependencies-job',a.http_python_dependencies_job,
            '--tokenizer',a.tokenizer,'--server-python',sys.executable,'--public-inputs',a.public_inputs,
            '--lock',a.lock,'--port',a.port,'--output',quality])
        state('independently-rescoring-full-Gemma-raw-quality')
        raw=a.output/'raw-quality-review.json'
        run('raw-quality-review','review_independent_full_task_quality.py',[
            '--quality-job',quality,'--public-inputs',a.public_inputs,'--native-job',a.native_job,
            '--full-model-profile','gemma4-48layers','--output',raw])
        q=json.loads(raw.read_text())
        assert q['complete'] and q['reviewed_shots']==20 and q['actual_response_count']==9936
        assert q['native_runtime_gate_SHA256']==sha(a.native_job/'native-gate.json')
        assert sha(parent_path)==parent_hash and verify_runtime(a.native_job,package)==native
        result=dict(full_profile='gemma4-48layers',full_controls_result_SHA256=parent_hash,
            matched_control_result_SHA256=sha(matched/'result.json'),
            matched_raw_review_SHA256=sha(matched/'same-PTE-native-independent-review.json'),
            raw_quality_review_SHA256=sha(raw),raw_full_quality_pass=q['full_task_raw_quality_pass'],
            positive_performance_signals=control['positive_performance_signals'],
            original_full_performance_signals=doc['positive_performance_signals'],
            original_evidence_preserved=True,full_model_qualified=False,main_candidate_accepted=False,
            remaining_gates=['full qualification receipt','same-GPU old/main performance and memory','20 Gemma main configurations'])
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('fresh-full-Gemma-native-ABBA-and-quality-collected',positive_performance_signals=result['positive_performance_signals'])
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
