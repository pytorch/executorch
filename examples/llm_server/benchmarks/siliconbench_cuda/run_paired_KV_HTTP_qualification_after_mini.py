"""Finish real HTTP/mini qualification after the paired-storage diagnostic.

Reuse completed GPU controls with their original raw hash closure. A failed
native/performance diagnostic blocks this workflow. It never exports or
deploys a full model or accepts main results.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from independent_session_full_native_binding import sha, verify_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mini-diagnostic-job','native-job','tokenizer','lock','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
    parser.add_argument('--Gemma-source-preparation-job',type=Path)
    parser.add_argument('--http-python-dependencies-job',type=Path)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--parent-kind',choices=['paired-KV-storage','workspace-preserving-contraction'],
                        default='paired-KV-storage')
    args = parser.parse_args()
    gemma = args.mini_profile == 'gemma4-first6'
    assert 1024 <= args.port <= 65535
    assert not gemma or (args.Gemma_source_preparation_job and args.http_python_dependencies_job)
    assert not args.output.exists(); args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    contraction = args.parent_kind=='workspace-preserving-contraction'
    parent_script = ('run_workspace_preserving_contraction_mini_diagnostic.py' if contraction
                     else 'run_paired_KV_storage_mini_diagnostic.py')
    parent_completed = args.parent_kind+'-mini-diagnostic-collected'

    def state(status,**extra):
        path = args.output/'progress.pending.json'
        path.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        path.replace(args.output/'progress.json')

    def run(label,script,tail):
        command = [sys.executable,'-B','-u',str(package/script),*map(str,tail)]
        (args.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
        with (args.output/(label+'.log')).open('x') as log:
            subprocess.run(command,check=True,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,timeout=14400)

    try:
        until = time.monotonic()+8*3600
        parent = args.mini_diagnostic_job
        while True:
            if not (parent/'progress.json').is_file():
                # Native/source checks precede the parent's first progress write.
                # Wait only for the exact detached controller in its launch receipt.
                launch=json.loads(parent.with_name(parent.name+'-deployment.json').read_text())
                handle=Path('/proc')/str(launch['controller_pid'])/'cmdline'
                actual=handle.read_bytes().split(bytes([0])) if handle.is_file() else []
                assert any(Path(v.decode()).name==parent_script for v in actual if v)
                assert b'--output' in actual and actual[actual.index(b'--output')+1]==os.fsencode(parent)
                assert b'--native-job' in actual and actual[actual.index(b'--native-job')+1]==os.fsencode(args.native_job)
                assert time.monotonic()<until,'Preserve confirmed-live native preflight after observation timeout'
                state('waiting-confirmed-live-mini-native-preflight',parent_PID=launch['controller_pid'])
                time.sleep(20)
                continue
            progress = json.loads((parent/'progress.json').read_text())
            if progress['status']==parent_completed:break
            assert progress['status']!='failed',progress
            handle = Path('/proc')/str(progress['controller_pid'])/'cmdline'
            actual = handle.read_bytes().split(bytes([0])) if handle.is_file() else []
            if not actual:
                refreshed = json.loads((parent/'progress.json').read_text())
                if refreshed['status']==parent_completed:break
            assert any(Path(v.decode()).name==parent_script for v in actual if v)
            assert b'--output' in actual and actual[actual.index(b'--output')+1]==os.fsencode(parent)
            assert time.monotonic()<until,'Preserve verified-live mini after observation timeout'
            state('waiting-confirmed-live-paired-mini',parent_PID=progress['controller_pid'],parent_status=progress['status'])
            time.sleep(20)
        result_path=parent/'result.json';parent_hash=sha(result_path)
        diagnostic=json.loads(result_path.read_text())
        if contraction:
            assert diagnostic['workspace_preserving_contraction_candidate']
            assert diagnostic['compared_to_previous_qualified_private_KV_runtime']
            assert diagnostic['private_KV_growth_slack_policy_unchanged'] and diagnostic['private_KV_contraction_trim_policy_unchanged']
        else:
            assert diagnostic['paired_KV_single_backing_candidate'] and diagnostic['compared_to_original_native_mini_SDK']
        assert diagnostic['observed_storage_and_graph_correctness_pass'] and diagnostic['initial_KV_capacity']==512
        assert diagnostic['mini_profile']==args.mini_profile
        for path,expected in diagnostic['raw_reviews_SHA256'].items():assert sha(path)==expected,path
        if diagnostic['positive_performance_signals']:
            (args.output/'result.json').write_text(json.dumps(dict(HTTP_collected=False,
                parent_result_SHA256=parent_hash,positive_performance_signals=diagnostic['positive_performance_signals'],
                full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
            state('mini-performance-gate-not-passed')
            return
        native=verify_runtime(args.native_job,package)
        if contraction:
            assert native['workspace_preserving_contraction_spec_SHA256']==sha(package/'independent_session_workspace_preserving_contraction_source.json')
        else:
            assert native['paired_KV_storage_spec_SHA256']==sha(package/'independent_session_paired_KV_storage_source.json')
        # Memory worker preparation expects the completed native controller.
        n=json.loads((args.native_job/'progress.json').read_text())
        assert n['status']=='native-GPU-tests-passed'
        handle=Path('/proc')/str(n['controller_pid'])/'cmdline'
        assert not handle.is_file() or os.fsencode(args.native_job) not in handle.read_bytes()
        controls=args.output/'mini-controls';controls.mkdir()
        for name,target in (('rewind-probe','rewind-probe'),('model-latency','matched-control/after'),
                            ('capacity-control','capacity-control'),('same-artifact-rewind','rewind'),
                            ('graph-lifecycle','graphs')):
            source=parent/target;assert source.is_dir();(controls/name).symlink_to(source.resolve())
        (controls/'actual-graph-review.json').symlink_to((parent/'graph-review.json').resolve())
        dependencies=args.http_python_dependencies_job
        if not gemma:
            state('preparing-paired-native-HTTP-dependencies')
            dependencies=args.output/'HTTP-dependencies'
            run('HTTP-dependencies','prepare_mg_http_python_dependencies.py',[
                '--native-job',args.native_job,'--python',sys.executable,'--output',dependencies])
        state('building-paired-native-memory-worker')
        run('memory-worker','prepare_gemma_kv_memory_worker.py' if gemma else 'prepare_kv_memory_worker.py',[
            '--native-job',args.native_job,'--python',sys.executable,'--output',controls/'memory-worker'])
        state('collecting-official-paired-mini-HTTP')
        tail=['--preparation-job',args.native_job/'same-artifact-mini-preparation',
              '--python',sys.executable,'--tokenizer',args.tokenizer,'--lock',args.lock,
              '--mini-profile',args.mini_profile,'--port',args.port,
              '--memory-worker-proof',controls/'memory-worker/proof.json',
              '--http-python-dependencies-job',dependencies,'--output',controls/'HTTP-memory']
        if gemma:tail+=['--Gemma-source-preparation-job',args.Gemma_source_preparation_job]
        run('HTTP-memory','run_independent_session_mini_http.py',tail)
        jobs={}
        for name in ('rewind-probe','model-latency','capacity-control','same-artifact-rewind',
                     'graph-lifecycle','memory-worker','HTTP-memory'):
            path=controls/name/('proof.json' if name in ('rewind-probe','memory-worker') else 'result.json')
            jobs[name]=dict(result=str(path),SHA256=sha(path))
        path=controls/'actual-graph-review.json';jobs['graph-review']=dict(result=str(path),SHA256=sha(path))
        (controls/'result.json').write_text(json.dumps(dict(jobs=jobs,commands={},
            native_gate_SHA256=sha(args.native_job/'native-gate.json'),
            same_compiled_model_artifacts=True,four_rounds_first_discarded=True,
            reused_completed_native_GPU_controls=True,preserved_native_controls_job=str(parent),
            parent_result_SHA256=parent_hash,controls_collected_pending_independent_review=True,
            full_model_qualified=False,candidate_accepted=False),indent=2)+'\n')
        (controls/'progress.json').write_text(json.dumps(dict(
            status='growth-allocator-mini-controls-collected-pending-independent-review',
            controller_pid=os.getpid(),reused_completed_native_GPU_controls=True))+'\n')
        state('independently-rechecking-complete-mini-qualification')
        tail=['--mini-job',controls,'--native-job',args.native_job,
              '--previous-latency-job',parent/'matched-control/before',
              '--matched-native-control-job',parent/'matched-control',
              '--preserved-native-controls-job',parent,'--preserved-native-controls-kind',args.parent_kind,
              '--profile',args.mini_profile,
              '--output',args.output/'mini-reviews']
        if gemma:tail+=['--Gemma-source-preparation-job',args.Gemma_source_preparation_job]
        run('qualification','qualify_independent_session_mini_for_full.py',tail)
        qualification=args.output/'mini-reviews/mini-qualification.json'
        q=json.loads(qualification.read_text())
        assert q['runtime_native_gate_SHA256']==sha(args.native_job/'native-gate.json')
        assert sha(result_path)==parent_hash and verify_runtime(args.native_job,package)==native
        (args.output/'result.json').write_text(json.dumps(dict(
            mini_qualification=str(qualification),mini_qualification_SHA256=sha(qualification),
            parent_result_SHA256=parent_hash,positive_performance_signals=q['positive_performance_signals'],
            full_model_export_for_qualification_allowed=q['full_model_export_for_qualification_allowed'],
            full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
        state('paired-mini-qualified-pending-full-model' if q['full_model_export_for_qualification_allowed']
              else 'mini-performance-gate-not-passed')
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
