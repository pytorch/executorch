"""Wait for the confirmed native build, then collect isolated same-PTE controls.

No full model export or acceptance. The original artifacts, SDK and prior raw
runs remain untouched. Every GPU collection takes the existing benchmark lock.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-job','output','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--nsys',type=Path,required=True)
    p.add_argument('--port',type=int,default=19892)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        temp=a.output/'progress.pending.json'
        temp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_accepted=False,full_model_export_started=False,**extra),indent=2)+'\n')
        temp.replace(a.output/'progress.json')
    def run(label,script,argv):
        state('running-growth-allocator-mini-control',control=label)
        command=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
        with (a.output/(label+'-controller.log')).open('x') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=7200,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        return command
    try:
        deadline=time.monotonic()+90*60
        alias=a.native_job/'same-artifact-mini-preparation'
        while True:
            old=json.loads((a.native_job/'progress.json').read_text())
            assert old['status']!='failed',old
            if old['status']=='native-GPU-tests-passed' and (alias/'progress.json').is_file():break
            pid=old['controller_pid'];handle=Path('/proc')/str(pid)/'cmdline'
            assert handle.is_file() and b'prepare_independent_session_growth_allocator.py' in handle.read_bytes()
            assert time.monotonic()<deadline,'Verified-live native observation timeout; do not restart build'
            state('waiting-on-confirmed-live-native-build',native_PID=pid,native_status=old['status'])
            time.sleep(15)
        gate=a.native_job/'native-gate.json';g=json.loads(gate.read_text())
        assert g['native_GPU_tests_pass'] and g['test_count']==74
        assert g['allocator_growth_spec_SHA256']==sha(package/'independent_session_growth_allocator_source.json')
        assert json.loads((alias/'progress.json').read_text())['original_compiled_artifacts_unchanged']
        jobs={};commands={}
        for label,script,argv in (
            ('model-latency','run_independent_session_mini_model_latency.py',
                ['--preparation-job',alias,'--output',a.output/'model-latency','--lock',a.lock,
                 '--mini-profile','muse-glimmer-first4','--native-sampler','off','--nsys',a.nsys]),
            ('capacity-control','run_independent_session_mini_capacity_control.py',
                ['--preparation-job',alias,'--output',a.output/'capacity-control','--lock',a.lock,
                 '--mini-profile','muse-glimmer-first4']),
            ('rewind-probe','prepare_rewind_kv_probe.py',
                ['--preparation-job',alias,'--python',sys.executable,'--output',a.output/'rewind-probe']),
            ('same-artifact-rewind','run_independent_session_same_artifact_control.py',
                ['--preparation-job',alias,'--output',a.output/'same-artifact-rewind','--lock',a.lock,
                 '--mini-profile','muse-glimmer-first4','--rewind-probe-job',a.output/'rewind-probe'])):
            commands[label]=run(label,script,argv)
            result=a.output/label/('proof.json' if label=='rewind-probe' else 'result.json')
            assert result.is_file();jobs[label]=dict(result=str(result),SHA256=sha(result))
        result=dict(controls_collected_pending_independent_review=True,commands=commands,jobs=jobs,
            native_gate_SHA256=sha(gate),same_compiled_model_artifacts=True,
            four_rounds_first_discarded=True,main_preset_changed=False,
            original_records_preserved=True,full_model_qualified=False,candidate_accepted=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('growth-allocator-mini-controls-collected-pending-independent-review')
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
