"""Run retained full Gemma gates only after this allocator's actual mini pass.

Reuse both immutable full artifacts and their original export provenance.
Preserve previous failures and timing signals; F1/main acceptance is separate.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_qualification, verify_runtime
from independent_session_mini_profiles import FULL_PROFILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mini-controller','native-job','preceding-native-job',
                 'preceding-full-rebind-proof','independent-export-controller',
                 'global-export-controller','Gemma-source-preparation-job',
                 'http-python-dependencies-job','tokenizer','preserved-global-rewind-failure-job',
                 'preserved-global-HTTP-failure-job','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
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
            subprocess.run(cmd,check=True,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,timeout=21600)
    try:
        qpath=a.mini_controller/'mini-reviews/mini-qualification.json'
        until=time.monotonic()+4*3600
        while not qpath.is_file():
            d=json.loads((a.mini_controller/'progress.json').read_text())
            assert d['status'] not in ('failed','mini-performance-gate-not-passed'),d
            proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
            actual=proc.read_bytes() if proc.is_file() else b''
            if not actual and qpath.is_file():break
            assert b'run_MG_allocator_mini_qualification_after_native.py' in actual
            assert str(a.mini_controller).encode() in actual
            assert time.monotonic()<until,'Preserve confirmed-live mini after observation timeout'
            state('waiting-confirmed-live-allocator-mini',mini_PID=d['controller_pid'],mini_status=d['status'])
            time.sleep(20)
        verify_qualification(qpath,'gemma4-first6',FULL_PROFILES['gemma4-48layers']['GGUF_SHA256'],a.native_job)
        verify_runtime(a.native_job,package)
        ind=a.independent_export_controller/'full-export'
        glob=a.global_export_controller/'full-export'
        rebind=a.output/'full-runtime-rebind';fixed=a.output/'full-fixed'
        lifecycle=a.output/'full-lifecycle';HTTP=a.output/'full-HTTP'
        state('rebinding-retained-full48-artifacts-after-mini-pass')
        run('rebind','prepare_independent_full_runtime_rebind.py',[
            '--package',package,'--export-runtime-job',a.preceding_native_job,'--runtime-job',a.native_job,
            '--mini-qualification',qpath,'--native-transition','workspace-preserving-growth-trim',
            '--preceding-full-rebind-proof',a.preceding_full_rebind_proof,
            '--independent-artifact',ind/'full-independent-artifact','--independent-export-job',ind,
            '--global-artifact',glob/'full-global-artifact','--global-export-job',glob,
            '--full-model-profile','gemma4-48layers','--output',rebind])
        state('collecting-full48-fixed-token-performance')
        run('fixed','run_full_independent_session_fixed_qualification.py',[
            '--package',package,'--native-job',a.native_job,'--runtime-rebind-job',rebind,
            '--independent-export-controller',a.independent_export_controller,
            '--global-export-controller',a.global_export_controller,'--full-model-profile','gemma4-48layers',
            '--lock',a.lock,'--nsys',a.nsys,'--output',fixed])
        state('collecting-full48-storage-rewind-graphs')
        run('lifecycle','run_full_independent_session_lifecycle_after_fixed.py',[
            '--package',package,'--native-job',a.native_job,'--fixed-job',fixed,
            '--full-model-profile','gemma4-48layers','--lock',a.lock,'--nsys',a.nsys,
            '--preserved-global-rewind-failure-job',a.preserved_global_rewind_failure_job,'--output',lifecycle])
        state('collecting-full48-independent-HTTP-memory')
        run('HTTP','run_gemma_full_http_after_lifecycle.py',[
            '--lifecycle-job',lifecycle,'--fixed-job',fixed,'--native-job',a.native_job,
            '--memory-worker-proof',a.mini_controller/'mini-controls/memory-worker/proof.json',
            '--Gemma-source-preparation-job',a.Gemma_source_preparation_job,
            '--http-python-dependencies-job',a.http_python_dependencies_job,
            '--tokenizer',a.tokenizer,'--lock',a.lock,'--port',a.port,
            '--preserved-global-HTTP-failure-job',a.preserved_global_HTTP_failure_job,'--output',HTTP])
        verify_runtime(a.native_job,package)
        results={name:json.loads((job/'result.json').read_text()) for name,job in
                 (('fixed',fixed),('lifecycle',lifecycle),('HTTP',HTTP))}
        result=dict(full_profile='gemma4-48layers',native_job=str(a.native_job),
            native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),
            mini_qualification_SHA256=sha(qpath),
            completed_result_SHA256={str(job/'result.json'):sha(job/'result.json') for job in (fixed,lifecycle,HTTP)},
            positive_performance_signals=results['HTTP']['positive_performance_signals'],
            full_model_qualified=False,main_candidate_accepted=False,
            remaining_gates=['full task F1','same-GPU old/new main performance and memory'])
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('full48-allocator-controls-collected-pending-F1-and-main',
              positive_performance_signals=result['positive_performance_signals'])
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
