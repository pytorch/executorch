"""Recover failed HTTP provenance and qualify preserved completed mini controls."""
import argparse, datetime, json, os, subprocess, sys, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prior-mini-controller','native-job','tokenizer','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
    p.add_argument('--Gemma-source-preparation-job',type=Path)
    p.add_argument('--http-python-dependencies-job',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args();package=Path(__file__).resolve().parent
    old=json.loads((a.prior_mini_controller/'progress.json').read_text());assert old['status']=='failed'
    handle=Path('/proc')/str(old['controller_pid'])/'cmdline'
    assert not handle.is_file() or str(a.prior_mini_controller).encode() not in handle.read_bytes()
    assert not a.output.exists();a.output.mkdir(parents=True)
    def state(status,**extra):
        f=a.output/'progress.pending.json'
        f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        f.replace(a.output/'progress.json')
    def run(label,script,argv):
        command=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(command,check=True,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,timeout=14400)
    try:
        gate=verify_runtime(a.native_job,package)
        state('recovering-only-missing-HTTP-and-graph-controls')
        common=['--Gemma-source-preparation-job',a.Gemma_source_preparation_job] if a.profile=='gemma4-first6' else []
        run('recovery','resume_independent_session_mini_after_HTTP_failure.py',[
            '--prior-mini-job',a.prior_mini_controller/'mini-controls','--native-job',a.native_job,
            '--tokenizer',a.tokenizer,'--lock',a.lock,'--nsys',a.nsys,'--profile',a.profile,
            '--http-python-dependencies-job',a.http_python_dependencies_job,
            '--port',a.port,'--output',a.output/'mini-controls',*common])
        state('independently-reviewing-preserved-and-recovered-mini-raw')
        run('qualification','qualify_independent_session_mini_for_full.py',[
            '--mini-job',a.output/'mini-controls','--native-job',a.native_job,
            '--previous-latency-job',a.prior_mini_controller/'original-latency',
            '--profile',a.profile,'--output',a.output/'mini-reviews',*common])
        qpath=a.output/'mini-reviews/mini-qualification.json';q=json.loads(qpath.read_text())
        assert q['runtime_native_gate_SHA256']==sha(a.native_job/'native-gate.json')
        assert q['mini_storage_correctness_pass'] and q['mini_HTTP_memory_pass']
        assert verify_runtime(a.native_job,package)==gate
        result=dict(mini_qualification=str(qpath),mini_qualification_SHA256=sha(qpath),
            mini_cold_and_steady_performance_pass=q['mini_cold_and_steady_performance_pass'],
            positive_performance_signals=q['positive_performance_signals'],
            full_model_export_for_qualification_allowed=q['full_model_export_for_qualification_allowed'],
            prior_failed_controller_preserved=str(a.prior_mini_controller),
            full_model_qualified=False,main_candidate_accepted=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('allocator-mini-qualified-pending-full-model' if q['full_model_export_for_qualification_allowed']
              else 'mini-performance-gate-not-passed',positive_performance_signals=q['positive_performance_signals'])
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
