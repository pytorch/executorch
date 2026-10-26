"""Resolve a consecutive native timing signal with matched four-round ABBA.

Preserve the failed receipt, all raw runs, the same kernels and thresholds.
Wait for the owned mini workflow to finish, then repeat only the native
comparison with per-process warmup and readonly GPU clock/power observations.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mini-controller','native-job','before-preparation',
                 'before-rewind-probe','previous-latency-job','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
    p.add_argument('--Gemma-source-preparation-job',type=Path)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
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
        qpath=a.mini_controller/'mini-reviews/mini-qualification.json'
        until=time.monotonic()+4*3600
        while not qpath.is_file():
            d=json.loads((a.mini_controller/'progress.json').read_text());assert d['status']!='failed',d
            proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
            actual=proc.read_bytes() if proc.is_file() else b''
            if not actual and qpath.is_file():break
            assert any(name in actual for name in (b'run_MG_allocator_mini_qualification_after_native.py',
                                                   b'resume_allocator_mini_after_HTTP_failure.py'))
            assert str(a.mini_controller).encode() in actual and time.monotonic()<until
            state('waiting-confirmed-live-mini-review',mini_PID=d['controller_pid'],mini_status=d['status'])
            time.sleep(20)
        q=json.loads(qpath.read_text());assert q['mini_profile']==a.profile
        original_receipt_SHA256=sha(qpath)
        assert q['mini_storage_correctness_pass'] and q['mini_HTTP_memory_pass']
        assert q['runtime_native_gate_SHA256']==sha(a.native_job/'native-gate.json')
        if q['mini_cold_and_steady_performance_pass']:
            state('mini-passed-no-extra-native-control-required');return
        assert q['positive_performance_signals'] and all(
            x['comparison']=='same-PTE-native-change' for x in q['positive_performance_signals']),q['positive_performance_signals']
        gate=verify_runtime(a.native_job,package)
        state('collecting-matched-native-ABBA',preserved_consecutive_signals=q['positive_performance_signals'])
        run('matched-control','run_interleaved_native_clock_control.py',[
            '--before-preparation',a.before_preparation,'--before-rewind-probe',a.before_rewind_probe,
            '--after-preparation',a.native_job/'same-artifact-mini-preparation',
            '--after-rewind-probe',a.mini_controller/'mini-controls/rewind-probe',
            '--mini-profile',a.profile,'--lock',a.lock,'--nsys',a.nsys,'--output',a.output/'matched-control'])
        (a.output/'mini-controls').symlink_to((a.mini_controller/'mini-controls').resolve())
        dependencies=a.mini_controller/'HTTP-dependencies'
        if dependencies.is_dir():(a.output/'HTTP-dependencies').symlink_to(dependencies.resolve())
        state('reviewing-mini-with-matched-native-reference')
        run('qualification','qualify_independent_session_mini_for_full.py',[
            '--mini-job',a.output/'mini-controls','--native-job',a.native_job,
            '--previous-latency-job',a.previous_latency_job,'--matched-native-control-job',a.output/'matched-control',
            '--profile',a.profile,'--output',a.output/'mini-reviews']+(
                ['--Gemma-source-preparation-job',a.Gemma_source_preparation_job] if a.profile=='gemma4-first6' else []))
        new_path=a.output/'mini-reviews/mini-qualification.json';new=json.loads(new_path.read_text())
        assert verify_runtime(a.native_job,package)==gate
        assert sha(qpath)==original_receipt_SHA256
        result=dict(mini_qualification=str(new_path),mini_qualification_SHA256=sha(new_path),
            mini_cold_and_steady_performance_pass=new['mini_cold_and_steady_performance_pass'],
            positive_performance_signals=new['positive_performance_signals'],
            original_consecutive_receipt=str(qpath),original_consecutive_receipt_SHA256=sha(qpath),
            full_model_export_for_qualification_allowed=new['full_model_export_for_qualification_allowed'],
            full_model_qualified=False,main_candidate_accepted=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('allocator-mini-qualified-pending-full-model' if new['full_model_export_for_qualification_allowed']
              else 'mini-performance-gate-not-passed',positive_performance_signals=new['positive_performance_signals'])
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
