"""Resume only failed HTTP and missing graph controls; retain completed raw runs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from independent_session_full_native_binding import sha, verify_runtime
from independent_session_mini_profiles import validate_mini_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prior-mini-job','native-job','tokenizer','lock','nsys','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=['muse-glimmer-first4','gemma4-first6'],required=True)
    parser.add_argument('--http-python-dependencies-job',type=Path)
    parser.add_argument('--Gemma-source-preparation-job',type=Path)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args();package=Path(__file__).resolve().parent
    prior=args.prior_mini_job
    old=json.loads((prior/'progress.json').read_text())
    assert old['status']=='failed' and 'run_independent_session_mini_http.py' in old['error']
    handle=Path('/proc')/str(old['controller_pid'])/'cmdline'
    assert not handle.exists() or str(prior).encode() not in handle.read_bytes()
    HTTP=prior/'HTTP-memory'
    failed=json.loads((HTTP/'progress.json').read_text()) if (HTTP/'progress.json').exists() else None
    if failed:
        assert failed['status']=='failed'
        handle=Path('/proc')/str(failed['controller_pid'])/'cmdline'
        assert not handle.exists() or str(HTTP).encode() not in handle.read_bytes()
    gate=verify_runtime(args.native_job,package)
    alias=args.native_job/'same-artifact-mini-preparation'
    assert json.loads((alias/'progress.json').read_text())['original_compiled_artifacts_unchanged']
    for arm in ('global','independent'):
        audit=json.loads((alias/('mini-'+arm+'-artifact')/'independent-mini-artifact-audit.json').read_text())
        validate_mini_audit(audit,args.profile,arm=='independent')
        for name,record in audit['artifacts'].items():assert sha(alias/('mini-'+arm+'-artifact')/name)==record['sha256']
    assert not args.output.exists();args.output.mkdir(parents=True)
    def state(status,**extra):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_accepted=False,full_model_export_started=False,**extra),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    jobs={};reused={};commands={}
    def run(label,script,argv):
        state('running-recovered-mini-control',control=label)
        command=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
        commands[label]=command
        with (args.output/(label+'-controller.log')).open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=7200,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    def reuse(label,filename):
        original=prior/label
        result=original/filename;assert result.is_file()
        doc=json.loads(result.read_text())
        if filename=='result.json':assert len(doc['rounds']) in (4,8)
        target=args.output/label;assert not target.exists();target.symlink_to(original)
        jobs[label]=dict(result=str(target/filename),SHA256=sha(result))
        reused[label]=dict(original=str(original),actual_result_SHA256=sha(result))
    try:
        state('verifying-terminal-failure-and-preserved-controls')
        for label in ('rewind-probe','model-latency','capacity-control','same-artifact-rewind','memory-worker'):
            reuse(label,'proof.json' if label in ('rewind-probe','memory-worker') else 'result.json')
        assert json.loads((args.output/'memory-worker/proof.json').read_text())['native_gate_SHA256']==sha(args.native_job/'native-gate.json')
        prior_graph=prior/'graph-lifecycle'
        if (prior_graph/'result.json').exists():
            reuse('graph-lifecycle','result.json')
            assert (prior_graph/'round-1/cuda-graphs.sqlite').is_file()
            graph=prior/'actual-graph-review.json';assert graph.is_file()
            (args.output/graph.name).symlink_to(graph)
            jobs['graph-review']=dict(result=str(args.output/graph.name),SHA256=sha(graph))
        HTTP_args=['--preparation-job',alias,'--python',sys.executable,'--tokenizer',args.tokenizer,
            '--lock',args.lock,'--output',args.output/'HTTP-memory','--port',args.port,
            '--mini-profile',args.profile,'--memory-worker-proof',args.output/'memory-worker/proof.json']
        if args.http_python_dependencies_job:
            HTTP_args+=['--http-python-dependencies-job',args.http_python_dependencies_job]
        if args.profile=='gemma4-first6':
            assert args.Gemma_source_preparation_job and args.http_python_dependencies_job
            HTTP_args+=['--Gemma-source-preparation-job',args.Gemma_source_preparation_job]
        run('HTTP-memory','run_independent_session_mini_http.py',HTTP_args)
        result=args.output/'HTTP-memory/result.json';assert result.is_file()
        jobs['HTTP-memory']=dict(result=str(result),SHA256=sha(result))
        if 'graph-lifecycle' not in jobs:
            artifact=alias/'mini-independent-artifact'
            audit=json.loads((artifact/'independent-mini-artifact-audit.json').read_text())
            run('graph-lifecycle','run_independent_session_mini_probe.py',[
                '--probe',alias/'model-build/muse_glimmer_runtime_kv_probe','--artifact',artifact,
                '--native-gate',alias/'native-gate.json',
                '--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
                '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
                '--arm','independent','--mini-profile',args.profile,'--automatic-reclaim',
                '--job-dir',args.output/'graph-lifecycle','--lock',args.lock,'--nsys',args.nsys])
            trace=args.output/'graph-lifecycle/round-1/cuda-graphs.nsys-rep'
            before=sha(trace);sqlite=trace.with_suffix('.sqlite');assert not sqlite.exists()
            with (args.output/'graph-sqlite-export.log').open('x') as log:
                subprocess.run([str(args.nsys),'export','--type=sqlite','--output='+str(sqlite),str(trace)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            assert sha(trace)==before
            run('graph-review','review_independent_session_mini_graph_trace.py',[
                '--runtime-job',args.output/'graph-lifecycle','--sqlite',sqlite,
                '--output',args.output/'actual-graph-review.json'])
            for label,path in (('graph-lifecycle',args.output/'graph-lifecycle/result.json'),
                               ('graph-review',args.output/'actual-graph-review.json')):
                jobs[label]=dict(result=str(path),SHA256=sha(path))
        for label,record in reused.items():
            filename='proof.json' if label in ('rewind-probe','memory-worker') else 'result.json'
            assert sha(Path(record['original'])/filename)==record['actual_result_SHA256']
        assert verify_runtime(args.native_job,package)==gate
        (args.output/'result.json').write_text(json.dumps(dict(
            controls_collected_pending_independent_review=True,jobs=jobs,commands=commands,
            native_gate_SHA256=sha(args.native_job/'native-gate.json'),
            same_compiled_model_artifacts=True,four_rounds_first_discarded=True,
            prior_failed_controller=str(prior),prior_failed_progress_SHA256=sha(prior/'progress.json'),
            reused_actual_completed_jobs=reused,original_records_preserved=True,
            main_preset_changed=False,full_model_qualified=False,candidate_accepted=False),indent=2)+'\n')
        state('growth-allocator-mini-controls-collected-pending-independent-review')
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
