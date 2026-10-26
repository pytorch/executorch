"""Collect full Gemma HTTP memory after actual storage/graph qualification.

Four fresh rounds compare the two retained full48-layer layouts. Main runs and
F1 acceptance remain separate; timing regressions stay visible in this result.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('lifecycle-job','fixed-job','native-job','memory-worker-proof',
                 'Gemma-source-preparation-job','http-python-dependencies-job',
                 'tokenizer','output','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--preserved-global-HTTP-failure-job',type=Path)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        f=a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        f.replace(a.output/'progress.json')
    def run(label,script,argv):
        cmd=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=cmd),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=21600)
    try:
        deadline=time.monotonic()+12*3600
        terminal='full-storage-rewind-graph-collected-pending-HTTP-F1-and-main'
        while True:
            d=json.loads((a.lifecycle_job/'progress.json').read_text())
            if d['status']==terminal:break
            assert d['status']!='failed',d
            proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
            assert proc.is_file() and any(name in proc.read_bytes() for name in (b'run_gemma_full_lifecycle_after_fixed.py',b'run_full_independent_session_lifecycle_after_fixed.py'))
            assert time.monotonic()<deadline,'Preserve verified-live work; observation deadline reached'
            state('waiting-confirmed-live-full-lifecycle',PID=d['controller_pid'],lifecycle_status=d['status'])
            time.sleep(20)
        verify_runtime(a.native_job,package)
        result=json.loads((a.lifecycle_job/'result.json').read_text())
        preserved_global_HTTP_failure=None
        if a.preserved_global_HTTP_failure_job:
            from independent_session_preserved_http_failure import verify_failed_global_HTTP
            preserved_global_HTTP_failure=verify_failed_global_HTTP(a.preserved_global_HTTP_failure_job,a.fixed_job/'full-pair')
        assert result['full_profile']=='gemma4-48layers' and result['full_storage_rewind_graph_pass']
        assert result['native_gate_SHA256']==sha(a.native_job/'native-gate.json')
        for name,expected in result['raw_reviews_SHA256'].items():
            assert sha(a.lifecycle_job/name)==expected,name
        state('collecting-full48-HTTP-memory',positive_performance_signals=result['positive_performance_signals'])
        run('HTTP-memory','run_independent_session_mini_http.py',[
            '--preparation-job',a.fixed_job/'full-pair','--python',sys.executable,
            '--tokenizer',a.tokenizer,'--lock',a.lock,'--output',a.output/'HTTP-memory',
            '--port',a.port,'--mini-profile','gemma4-first6','--full-model-profile','gemma4-48layers',
            '--full-native-job',a.native_job,'--memory-worker-proof',a.memory_worker_proof,
            '--Gemma-source-preparation-job',a.Gemma_source_preparation_job,
            '--http-python-dependencies-job',a.http_python_dependencies_job]+(
            ['--arm','independent','--preserved-global-HTTP-failure-job',a.preserved_global_HTTP_failure_job] if preserved_global_HTTP_failure else []))
        run('HTTP-review','review_independent_session_http_lifecycle.py',[
            '--job',a.output/'HTTP-memory','--require-native-memory',
            '--output',a.output/'HTTP-independent-review.json'])
        review=json.loads((a.output/'HTTP-independent-review.json').read_text())
        assert review['full_model_profile']=='gemma4-48layers'
        if preserved_global_HTTP_failure:
            assert review['collected_arms']==['independent'] and review['matched_requests']==0
            assert review['actual_requests']==256 and review['server_runs']==8
            assert review['completion_lengths_equal_between_arms'] is None
        else:
            assert review['matched_requests']==256
        assert review['native_allocator_memory_verified']
        assert review['automatic_OFF_c16_to_c1_reclaim_verified']
        final=dict(full_profile='gemma4-48layers',full_HTTP_memory_pass=True,
            fixed_token_performance_pass=result['fixed_token_performance_pass'],
            positive_performance_signals=result['positive_performance_signals'],
            HTTP_review_SHA256=sha(a.output/'HTTP-independent-review.json'),
            lifecycle_result_SHA256=sha(a.lifecycle_job/'result.json'),
            natural_HTTP_outputs_are_not_task_F1_acceptance=True,
            rewind_test_arms=result.get('rewind_test_arms',['global','independent']),
            preserved_global_rewind_failure=result.get('preserved_global_rewind_failure'),
            HTTP_test_arms=review['collected_arms'],preserved_global_HTTP_failure=preserved_global_HTTP_failure,
            global_HTTP_pass=not bool(preserved_global_HTTP_failure),
            full_model_qualified=False,main_candidate_accepted=False,
            remaining_gates=['full 0/5-shot F1 and balanced checks','main comparison'])
        (a.output/'result.json').write_text(json.dumps(final,indent=2)+'\n')
        state('full-HTTP-memory-collected-pending-F1-and-main',positive_performance_signals=final['positive_performance_signals'])
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
