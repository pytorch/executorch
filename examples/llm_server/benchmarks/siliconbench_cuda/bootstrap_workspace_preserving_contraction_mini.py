"""Queue an isolated runtime/mini experiment after the existing measurements.

Wait for the actual native chat controller and, optionally, both owned Nsight
profiles. Do not overlap source copying/CPU builds with measured serving. Reuse
the already qualified private-KV SDK and byte-identical compiled minis. Require
74 GPU tests, four-round ABBA, storage/graph correctness and real HTTP memory.
This controller never starts a full-model export or modifies a main server.
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


def command(pid):
    path = Path('/proc')/str(pid)/'cmdline'
    stat = Path('/proc')/str(pid)/'stat'
    if not path.is_file() or not stat.is_file() or stat.read_text().rsplit(') ',1)[1][0]=='Z':
        return []
    return [x.decode() for x in path.read_bytes().split(bytes([0])) if x]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['after-chat-job','baseline-native-job','before-preparation','before-rewind-probe',
                 'mini-qualification','tokenizer','lock','nsys','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--after-profile-job',type=Path)
    p.add_argument('--mini-profile',choices=['muse-glimmer-first4','gemma4-first6'],required=True)
    p.add_argument('--Gemma-source-preparation-job',type=Path)
    p.add_argument('--http-python-dependencies-job',type=Path)
    p.add_argument('--expected-GPU-UUID',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--jobs',type=int,choices=[1,2,4],default=4)
    a = p.parse_args()
    assert 1024<=a.port<=65535
    gemma=a.mini_profile=='gemma4-first6'
    assert not gemma or (a.Gemma_source_preparation_job and a.http_python_dependencies_job)
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent

    def state(status,**fields):
        doc = dict(status=status,controller_pid=os.getpid(),
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   full_model_export_started=False,full_model_qualified=False,
                   main_candidate_accepted=False,baseline_source_artifacts_and_results_modified=False,**fields)
        path=a.output/'progress.pending.json'
        path.write_text(json.dumps(doc,indent=2)+'\n')
        path.replace(a.output/'progress.json')
        with (a.output/'history.jsonl').open('a') as log:log.write(json.dumps(doc)+'\n')

    def run(label,script,tail):
        argv=[sys.executable,'-B','-u',str(package/script),*map(str,tail)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(argv,check=True,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,timeout=8*3600)

    def idle():
        return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                           '--format=csv,noheader'],text=True).strip()

    try:
        deadline=time.monotonic()+6*3600
        while True:
            d=json.loads((a.after_chat_job/'progress.json').read_text())
            if d['status']=='corrected-native-group32-chat-collected-pending-independent-review':
                assert len(d['completed'])==14
                break
            assert d['status']!='failed',d
            cmd=command(d['controller_pid'])
            assert any(Path(x).name=='bootstrap_corrected_native_MG_chat.py' for x in cmd)
            assert '--output' in cmd and Path(cmd[cmd.index('--output')+1]).resolve()==a.after_chat_job.resolve()
            assert time.monotonic()<deadline,'Observation deadline only; original task preserved'
            state('waiting-confirmed-live-native-chat-no-build',chat_PID=d['controller_pid'],
                  chat_status=d['status'],chat_completed_cases=len(d.get('completed',[])))
            time.sleep(30)
        profile_receipts=[]
        if a.after_profile_job:
            while True:
                ready=True
                profile_receipts=[]
                for prefix in ['off','on']:
                    root=a.after_profile_job/('prefix-'+prefix)
                    deployment=root/'profile-deployment.json'
                    if not deployment.is_file():
                        ready=False
                        continue
                    launch=json.loads(deployment.read_text())
                    cmd=command(launch['PID'])
                    if cmd:
                        assert any('nsys' in x for x in cmd) and any(str(root) in x for x in cmd)
                        ready=False
                        state('waiting-confirmed-live-owned-profiler-no-build',prefix=prefix,profiler_PID=launch['PID'])
                        continue
                    required=[root/'trace.nsys-rep',root/'trace.sqlite']
                    for repetition in range(1,5):
                        required += [root/f'repeat-{repetition}/metrics.json',
                                     root/f'repeat-{repetition}/client-boundary-clocks.json']
                    if not all(x.is_file() and x.stat().st_size for x in required):
                        ready=False
                        continue
                    for repetition in range(1,5):
                        metrics=json.loads((root/f'repeat-{repetition}/metrics.json').read_text())['concurrency_results'][0]
                        assert metrics['successful']==32 and metrics['failed']==0
                    profile_receipts.append(dict(prefix=prefix,profile_PID=launch['PID'],
                        deployment_SHA256=sha(deployment),SQLite_SHA256=sha(root/'trace.sqlite'),
                        four_rounds_complete=True))
                if ready and len(profile_receipts)==2 and idle():
                    break
                assert time.monotonic()<deadline,'Expected profile not complete; preserve queue and original tasks'
                state('waiting-for-both-owned-profile-finalizations-no-build',completed_profiles=len(profile_receipts))
                time.sleep(20)
        while not idle():
            assert time.monotonic()<deadline,'GPU still active; no build or experiment started'
            state('waiting-original-GPU-processes-to-finish-no-build')
            time.sleep(10)
        GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,power.limit',
                                     '--format=csv,noheader'],text=True).strip()
        assert GPU.startswith(a.expected_GPU_UUID+',')
        q=json.loads(a.mini_qualification.read_text())
        for key in ['mini_storage_correctness_pass','mini_cold_and_steady_performance_pass','mini_HTTP_memory_pass']:
            assert q[key]
        assert q['mini_profile']==a.mini_profile
        assert q['runtime_native_gate_SHA256']==sha(a.baseline_native_job/'native-gate.json')
        baseline=verify_runtime(a.baseline_native_job,package)
        assert baseline.get('isolated_KV_pool_spec_SHA256')
        baseline_gate_hash=sha(a.baseline_native_job/'native-gate.json')
        inputs=dict(GPU=GPU,after_chat_job=str(a.after_chat_job),after_profile_job=str(a.after_profile_job),
                    completed_profiles=profile_receipts,baseline_native_gate_SHA256=baseline_gate_hash,
                    mini_qualification_SHA256=sha(a.mini_qualification),mini_profile=a.mini_profile,
                    isolated_shadow_runtime=True,full_export_started=False)
        (a.output/'provenance.json').write_text(json.dumps(inputs,indent=2)+'\n')
        native=a.output/'candidate-native'
        state('building-isolated-private-KV-contraction-candidate',GPU=GPU)
        run('native','prepare_independent_session_coalesced_growth_trim.py',[
            '--package',package,'--baseline-native-job',a.baseline_native_job,
            '--source-variant','Gemma-empty-owner-reuse' if gemma else 'MG-empty-owner-reuse',
            '--allocator-policy','workspace-preserving-contraction','--python',sys.executable,
            '--mini-preparation-job',a.before_preparation,'--mini-profile',a.mini_profile,
            '--lock',a.lock,'--jobs',a.jobs,'--output',native])
        after=verify_runtime(native,package)
        assert after['preserved_preceding_native_gate_SHA256']==baseline_gate_hash
        state('collecting-same-PTE-native-mini-controls')
        mini=a.output/'mini-diagnostic'
        run('mini','run_workspace_preserving_contraction_mini_diagnostic.py',[
            '--native-job',native,'--baseline-native-job',a.baseline_native_job,
            '--before-preparation',a.before_preparation,'--before-rewind-probe',a.before_rewind_probe,
            '--mini-profile',a.mini_profile,'--lock',a.lock,'--nsys',a.nsys,'--output',mini])
        diagnostic=json.loads((mini/'result.json').read_text())
        if diagnostic['positive_performance_signals']:
            state('mini-performance-signal-kept-no-full-deployment',signals=diagnostic['positive_performance_signals'])
            return
        state('collecting-real-mini-HTTP-memory-and-reclaim')
        tail=['--mini-diagnostic-job',mini,'--native-job',native,'--tokenizer',a.tokenizer,
              '--lock',a.lock,'--port',a.port,'--mini-profile',a.mini_profile,
              '--parent-kind','workspace-preserving-contraction','--output',a.output/'HTTP-qualification']
        if gemma:tail+=['--Gemma-source-preparation-job',a.Gemma_source_preparation_job,
                       '--http-python-dependencies-job',a.http_python_dependencies_job]
        run('HTTP','run_paired_KV_HTTP_qualification_after_mini.py',tail)
        result=json.loads((a.output/'HTTP-qualification/result.json').read_text())
        assert sha(a.baseline_native_job/'native-gate.json')==baseline_gate_hash
        assert verify_runtime(a.baseline_native_job,package)==baseline and verify_runtime(native,package)==after
        final=dict(inputs,candidate_native_gate_SHA256=sha(native/'native-gate.json'),
                   mini_qualification=result['mini_qualification'],
                   mini_qualification_SHA256=result['mini_qualification_SHA256'],
                   full_model_export_for_qualification_allowed=result['full_model_export_for_qualification_allowed'],
                   positive_performance_signals=result['positive_performance_signals'],
                   main_candidate_accepted=False,full_model_qualified=False,full_model_export_started=False)
        (a.output/'result.json').write_text(json.dumps(final,indent=2)+'\n')
        state('mini-qualified-pending-full-correctness-and-performance' if final['full_model_export_for_qualification_allowed']
              else 'mini-gate-not-passed-no-full-deployment')
    except BaseException as e:
        state('failed-preserve-original-experiments',error=str(e),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
