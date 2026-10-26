"""Isolate growth-cache policy from the later native metadata/reuse repairs.

Build in an owned directory, require74 real GPU tests, then reuse identical
mini PTEs and matched probes for four-round ABBA diagnostics. No full export,
kernel tuning, hardware-setting change, or main acceptance occurs here.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-native-job','original-mini-preparation','original-rewind-probe',
                 'python','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        t=a.output/'progress.pending.json'
        t.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_promoted=False,full_model_qualified=False,**extra),indent=2)+'\n')
        t.replace(a.output/'progress.json')
    def run(label,script,args):
        command=[str(a.python),'-B','-u',str(package/script),*map(str,args)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=14400)
    try:
        baseline=json.loads((a.baseline_native_job/'native-gate.json').read_text())
        assert baseline['native_GPU_tests_pass'] and baseline['test_count']==74
        native=a.output/'growth-only-native'
        state('building-growth-policy-only-native')
        run('native','prepare_independent_session_growth_allocator.py',[
            '--baseline-source',baseline['source'],'--baseline-native-gate',a.baseline_native_job/'native-gate.json',
            '--python',a.python,'--mini-preparation-job',a.original_mini_preparation,
            '--lock',a.lock,'--output',native])
        gate=json.loads((native/'native-gate.json').read_text())
        assert gate['test_count']==74 and gate['native_GPU_tests_pass']
        assert not gate.get('stable_fastpath_spec_SHA256')
        assert not gate.get('stable_metadata_uploads_spec_SHA256') and not gate.get('flat_migration_copy_spec_SHA256')
        prep=native/'same-artifact-mini-preparation';probe=a.output/'growth-only-rewind-probe'
        state('building-matched-growth-policy-rewind-probe')
        run('rewind-probe','prepare_rewind_kv_probe.py',[
            '--preparation-job',prep,'--python',a.python,'--output',probe])
        state('collecting-original-versus-growth-only-ABBA')
        run('ABBA','run_interleaved_native_clock_control.py',[
            '--before-preparation',a.original_mini_preparation,'--before-rewind-probe',a.original_rewind_probe,
            '--after-preparation',prep,'--after-rewind-probe',probe,
            '--lock',a.lock,'--nsys',a.nsys,'--output',a.output/'ABBA'])
        from independent_session_full_native_binding import sha
        result=json.loads((a.output/'ABBA/result.json').read_text())
        assert result['unchanged_PTE_PTD_and_tokens'] and result['matched_probe_CPP']
        (a.output/'result.json').write_text(json.dumps(dict(
            actual_growth_policy_only=True,native_test_count=74,
            ABBA_result_SHA256=sha(a.output/'ABBA/result.json'),
            positive_performance_signals=result['positive_performance_signals'],
            previous_failed_gate_not_replaced=True,full_model_qualified=False,
            main_candidate_accepted=False),indent=2)+'\n')
        state('growth-only-native-stage-diagnostic-collected',positive_performance_signals=result['positive_performance_signals'])
    except Exception as e:
        state('failed',error=str(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
