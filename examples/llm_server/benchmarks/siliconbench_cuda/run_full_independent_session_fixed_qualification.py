"""Collect full52/48-layer fixed-token controls after both gated exports finish.

This is a same-GPU global/independent layout control, not an accepted main result
or a replacement for full task F1, lifecycle memory and HTTP measurements.
The actual full geometry, provenance and raw numerical differences are retained.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import verify_qualification, verify_runtime, alias_runtime, sha
from independent_session_mini_profiles import FULL_PROFILES, validate_full_audit
from prepare_independent_full_runtime_rebind import verify_rebind


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('independent-export-controller','global-export-controller','native-job','output','lock','nsys'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--full-model-profile',choices=FULL_PROFILES,required=True)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--runtime-rebind-job',type=Path,help='Recheck qualified native replacement and retain both existing full PTEs')
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=a.package.resolve()
    gemma=a.full_model_profile=='gemma4-48layers'
    mini_profile='gemma4-first6' if gemma else 'muse-glimmer-first4'
    GGUF_SHA256=FULL_PROFILES[a.full_model_profile]['GGUF_SHA256']
    def state(status,**extra):
        temporary=a.output/'progress.pending.json'
        temporary.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_model_qualified=False,
            main_candidate_accepted=False,**extra),indent=2)+'\n')
        temporary.replace(a.output/'progress.json')
    def run(label,script,argv):
        command=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           check=True,timeout=21600)
    try:
        roots={'independent':a.independent_export_controller,'global':a.global_export_controller}
        deadline=time.monotonic()+8*3600
        while True:
            pending=[]
            for arm,root in roots.items():
                d=json.loads((root/'progress.json').read_text())
                assert d['status'] not in ('failed','mini-performance-gate-not-passed'),d
                if d['status']=='full-artifact-ready-pending-full-model-qualification':continue
                proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
                assert proc.is_file() and b'run_independent_full_after_mini.py' in proc.read_bytes()
                pending.append(dict(arm=arm,PID=d['controller_pid'],status=d['status']))
            if not pending:break
            assert time.monotonic()<deadline,'Verified-live export observation expired; preserve existing work'
            state('waiting-confirmed-live-full-exports',jobs=pending)
            time.sleep(20)
        gate=verify_runtime(a.native_job,package)
        rebind=None
        if a.runtime_rebind_job:
            proof_path=a.runtime_rebind_job/'full-runtime-rebind.json'
            rebind=verify_rebind(proof_path,a.native_job,package)
            assert rebind['full_model_profile']==a.full_model_profile
            assert set(rebind['artifacts'])=={'global','independent'}
            qpath=Path(rebind['mini_qualification'])
            qualification=verify_qualification(qpath,mini_profile,GGUF_SHA256,a.native_job)
            audit_hashes={};artifacts={}
            for arm,item in rebind['artifacts'].items():
                artifact=Path(item['artifact']);audit_path=artifact/'independent-full-artifact-audit.json'
                audit=json.loads(audit_path.read_text());validate_full_audit(audit,a.full_model_profile,arm=='independent')
                for name,r in audit['artifacts'].items():assert sha(artifact/name)==r['sha256'],name
                artifacts[arm]=artifact;audit_hashes[arm]=sha(audit_path)
            pair=a.output/'full-pair';pair.mkdir()
            alias_runtime(a.native_job,package,pair,verify_runtime(Path(rebind.get('original_model_export_runtime_job', rebind['export_runtime_job'])),package)['source'],extra={
                'full_runtime_rebind_proof':str(proof_path.resolve()),'full_runtime_rebind_proof_SHA256':sha(proof_path)})
            for arm,artifact in artifacts.items():(pair/('full-'+arm+'-artifact')).symlink_to(artifact)
            (pair/'progress.json').write_text(json.dumps(dict(status='full-artifacts-ready-for-qualification',
                controller_pid=os.getpid(),mini_gate_passed=True,all_full_artifact_hashes_verified=True,
                full_model_qualified=False,full_profile=a.full_model_profile,full_audit_SHA256=audit_hashes,
                full_runtime_rebind_proof_SHA256=sha(proof_path)),indent=2)+'\n')
        else:
            artifacts={};audit_hashes={};qualifications={}
            for arm,root in roots.items():
                qpath=root/'mini-reviews/mini-qualification.json'
                qualifications[arm]=verify_qualification(qpath,mini_profile,GGUF_SHA256,a.native_job)
                export_state=json.loads((root/'progress.json').read_text())
                artifact=Path(export_state['artifact'])
                assert artifact.resolve().is_relative_to(root.resolve())
                audit_path=artifact/'independent-full-artifact-audit.json';audit=json.loads(audit_path.read_text())
                validate_full_audit(audit,a.full_model_profile,arm=='independent')
                assert audit['native_gate_sha256']==sha(root/'full-export/native-gate.json')
                assert audit['bulk_prefill_A16_original_native_decode']
                if gemma:
                    assert audit['precision_kernel_sha256']=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
                    assert audit['safe_ring_rows']==1535
                for name,r in audit['artifacts'].items():assert sha(artifact/name)==r['sha256'],name
                artifacts[arm]=artifact;audit_hashes[arm]=sha(audit_path)
            if gemma:
                assert qualifications['global']['Gemma_source_preparation_SHA256']==qualifications['independent']['Gemma_source_preparation_SHA256']
            pair=a.output/'full-pair';pair.mkdir()
            alias_runtime(a.native_job,package,pair,gate['source'])
            for arm,artifact in artifacts.items():(pair/('full-'+arm+'-artifact')).symlink_to(artifact)
            (pair/'progress.json').write_text(json.dumps(dict(status='full-artifacts-ready-for-qualification',
                controller_pid=os.getpid(),mini_gate_passed=True,all_full_artifact_hashes_verified=True,
                full_model_qualified=False,full_profile=a.full_model_profile,full_audit_SHA256=audit_hashes),indent=2)+'\n')
        state('building-full-control-probe')
        run('rewind-probe','prepare_rewind_kv_probe.py',['--preparation-job',pair,'--python',sys.executable,'--output',a.output/'rewind-probe'])
        state('collecting-full-fixed-token-latency',full_profile=a.full_model_profile)
        run('model-latency','run_independent_session_mini_model_latency.py',[
            '--preparation-job',pair,'--output',a.output/'model-latency','--lock',a.lock,
            '--mini-profile',mini_profile,'--full-model-profile',a.full_model_profile,
            '--native-sampler','off','--nsys',a.nsys,'--rewind-probe-job',a.output/'rewind-probe'])
        reviewer='review_independent_session_'+('gemma' if gemma else 'mg')+'_fixed_latency.py'
        run('fixed-latency-review',reviewer,[
            '--job',a.output/'model-latency','--output',a.output/'fixed-latency-independent-review.json'])
        review=json.loads((a.output/'fixed-latency-independent-review.json').read_text());signals=[]
        for cell in review['aggregates']:
            for metric in ('cold_first_four_seconds','mean_stable_native_ms'):
                r=cell[metric];delta=r['arms']['independent']['mean']-r['arms']['global']['mean']
                if delta>r['combined_3SE']:signals.append(dict(case=cell['case'],metric=metric,delta=delta,
                    combined_3SE=r['combined_3SE'],delta_percent=r['delta_percent']))
        (a.output/'result.json').write_text(json.dumps(dict(full_profile=a.full_model_profile,
            full_audit_SHA256=audit_hashes,runtime_rebind_proof_SHA256=sha(a.runtime_rebind_job/'full-runtime-rebind.json') if rebind else None,raw_latency_result_SHA256=sha(a.output/'model-latency/result.json'),
            independent_review_SHA256=sha(a.output/'fixed-latency-independent-review.json'),
            positive_performance_signals=signals,fixed_token_performance_pass=not signals,
            remaining_gates=['full task F1','same-artifact lifecycle and rewind','actual post-growth graph replay','HTTP overhead and sampled allocator memory','main rerun'],
            fresh_global_export_is_a_layout_control_not_the_previous_accepted_PTE=True,
            full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
        state('full-fixed-token-collected-pending-other-full-gates',positive_performance_signals=signals)
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
