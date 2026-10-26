"""Verify a coalesced growth-trim policy on identical compiled MG mini graphs.

Compare the candidate with both the original SDK and the currently slower SDK
using four-round ABBA controls. Then verify allocation, rewind and actual CUDA
replay. Preserve every prior attempt; do not deploy or accept a full model.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-native-job','original-mini-preparation','original-rewind-probe',
                 'current-mini-preparation','current-rewind-probe','python','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allocator-policy',choices=('coalesced','pool-only'),default='coalesced')
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
        baseline=verify_runtime(a.baseline_native_job,package)
        assert baseline.get('empty_owner_reuse_spec_SHA256')
        native=a.output/('pool-only-trim-native' if a.allocator_policy=='pool-only' else 'coalesced-trim-native')
        state('building-coalesced-growth-trim-native')
        run('native','prepare_independent_session_coalesced_growth_trim.py',[
            '--package',package,'--baseline-native-job',a.baseline_native_job,
            '--source-variant','MG-empty-owner-reuse','--mini-profile','muse-glimmer-first4',
            '--allocator-policy',a.allocator_policy,
            '--python',a.python,'--mini-preparation-job',a.original_mini_preparation,
            '--lock',a.lock,'--output',native])
        gate=verify_runtime(native,package)
        assert gate['test_count']==74 and gate['native_GPU_tests_pass']
        assert gate['empty_owner_reuse_spec_SHA256']==baseline['empty_owner_reuse_spec_SHA256']
        prep=native/'same-artifact-mini-preparation';probe=a.output/'coalesced-trim-rewind-probe'
        state('building-matched-coalesced-trim-rewind-probe')
        run('rewind-probe','prepare_rewind_kv_probe.py',[
            '--preparation-job',prep,'--python',a.python,'--output',probe])
        comparisons=[]
        for label,before,before_probe in (
            ('original-SDK-ABBA',a.original_mini_preparation,a.original_rewind_probe),
            ('current-SDK-ABBA',a.current_mini_preparation,a.current_rewind_probe)):
            state('collecting-coalesced-trim-ABBA',comparison=label)
            run(label,'run_interleaved_native_clock_control.py',[
                '--before-preparation',before,'--before-rewind-probe',before_probe,
                '--after-preparation',prep,'--after-rewind-probe',probe,
                '--lock',a.lock,'--nsys',a.nsys,'--output',a.output/label])
            result=json.loads((a.output/label/'result.json').read_text())
            assert result['unchanged_PTE_PTD_and_tokens'] and result['matched_probe_CPP']
            comparisons.append(dict(comparison=label,result_SHA256=sha(a.output/label/'result.json'),
                positive_performance_signals=result['positive_performance_signals']))
        state('verifying-coalesced-trim-mini-capacity-rewind-graphs')
        run('mini-controls','run_growth_allocator_mg_mini_controls.py',[
            '--native-job',native,'--lock',a.lock,'--nsys',a.nsys,'--output',a.output/'mini-controls'])
        controls=json.loads((a.output/'mini-controls/result.json').read_text())
        assert controls['same_compiled_model_artifacts'] and controls['four_rounds_first_discarded']
        verify_runtime(native,package);assert verify_runtime(a.baseline_native_job,package)==baseline
        (a.output/'result.json').write_text(json.dumps(dict(
            native_test_count=74,allocator_policy=a.allocator_policy,comparisons=comparisons,
            mini_controls_result_SHA256=sha(a.output/'mini-controls/result.json'),
            previous_failed_gate_not_replaced=True,full_model_qualified=False,
            main_candidate_accepted=False,
            scope='Allocator policy candidate diagnostic; independent raw review, full mini HTTP and full-model gates remain required.'),indent=2)+'\n')
        state('coalesced-growth-trim-mini-diagnostic-collected',comparisons=comparisons)
    except Exception as e:
        state('failed',error=str(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
