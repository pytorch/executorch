"""Qualify a newly exported MG PTE with same-PTE native and storage controls.

Wait for the existing gated exporter, preserve all artifacts, verify exact
source/weights/runtime chains, and collect candidate-only full-model evidence.
No separately compiled global result, full F1, or main acceptance is claimed.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime,verify_qualification,alias_runtime
from independent_session_mini_profiles import FULL_PROFILES,validate_full_audit


def verify_original_export_native_gate(original_path, exported_path):
    """Accept direct identity or an exact hash-bound binary-path alias expansion."""
    original_path, exported_path = Path(original_path), Path(exported_path)
    original = json.loads(original_path.read_text())
    exported = json.loads(exported_path.read_text())
    aliases = {}
    if exported != original:
        assert exported['imported_native_gate_SHA256'] == sha(original_path)
        aliases = exported['verified_binary_path_aliases']
        assert isinstance(aliases, dict) and aliases
        binaries = dict(original['binaries'])
        for alias, target in aliases.items():
            assert alias not in binaries and target in original['binaries']
            assert Path(alias).resolve() == Path(target).resolve()
            binaries[alias] = original['binaries'][target]
        expected = {**original, 'binaries': binaries,
                    'imported_native_gate_SHA256': sha(original_path),
                    'verified_binary_path_aliases': aliases}
        assert exported == expected, 'Unexpected original SDK provenance change'
    for name, expected in exported['binaries'].items():
        assert sha(name) == expected, name
    return {'original_native_gate_SHA256': sha(original_path),
            'export_original_native_gate_SHA256': sha(exported_path),
            'exact_metadata_and_binary_hash_binding_verified': True,
            'verified_additional_binary_aliases': aliases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export-controller','original-native-job','native-job','lock','nsys','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent;profile='muse-glimmer-52layers'
    def state(status,**extra):
        t=a.output/'progress.pending.json'
        t.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        t.replace(a.output/'progress.json')
    def run(label,script,args):
        cmd=[sys.executable,'-B','-u',str(package/script),*map(str,args)]
        (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=cmd),indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=21600)
    try:
        deadline=time.monotonic()+12*3600
        while True:
            d=json.loads((a.export_controller/'progress.json').read_text())
            assert d['status'] not in ('failed','mini-performance-gate-not-passed'),d
            if d['status']=='full-artifact-ready-pending-full-model-qualification':break
            handle=Path('/proc')/str(d['controller_pid'])/'cmdline'
            assert handle.is_file() and b'run_independent_full_after_mini.py' in handle.read_bytes()
            assert time.monotonic()<deadline,'Preserve verified-live exporter after observation deadline'
            state('waiting-confirmed-live-full-MG-export',PID=d['controller_pid'],export_status=d['status'])
            time.sleep(20)
        qpath=a.export_controller/'mini-reviews/mini-qualification.json'
        verify_qualification(qpath,'muse-glimmer-first4',FULL_PROFILES[profile]['GGUF_SHA256'],a.native_job)
        after=verify_runtime(a.native_job,package)
        artifact=Path(d['artifact']);assert artifact.resolve().is_relative_to(a.export_controller.resolve())
        audit_path=artifact/'independent-full-artifact-audit.json';audit=json.loads(audit_path.read_text())
        validate_full_audit(audit,profile,True)
        export_job=a.export_controller/'full-export'
        assert audit['native_gate_sha256']==sha(export_job/'native-gate.json')
        export_gate=json.loads((export_job/'native-gate.json').read_text())
        for key in ('source','source_spec_SHA256','actual_native_source_change','unchanged_controls'):
            assert export_gate[key]==after[key],key
        for name,expected in export_gate['binaries'].items():assert sha(name)==expected,name
        for name,record in audit['artifacts'].items():assert sha(artifact/name)==record['sha256'],name
        assert audit['bulk_prefill_A16_original_native_decode']
        original=json.loads((a.original_native_job/'native-gate.json').read_text())
        assert original['native_GPU_tests_pass'] and original['test_count']==74
        assert not original.get('actual_native_source_change')
        spec_path=package/'independent_session_capacity_source.json';spec=json.loads(spec_path.read_text())
        assert sha(spec_path)==original['source_spec_SHA256']==after['source_spec_SHA256']
        assert sha(a.original_native_job/'native-tests.xml')==original['native_tests_xml_SHA256']
        tests=list(ET.parse(a.original_native_job/'native-tests.xml').getroot().iter('testcase'))
        assert len(tests)==74 and not any(t.find(x) is not None for t in tests for x in ('failure','error','skipped'))
        for name,record in spec['files'].items():assert sha(Path(original['source'])/name)==record['after_SHA256'],name
        for name,expected in original['unchanged_controls'].items():
            assert sha(Path(original['source'])/name)==expected and after['unchanged_controls'][name]==expected,name
        for name,expected in original['binaries'].items():assert sha(name)==expected,name
        assert audit['export_source_native_gate_sha256'] == sha(export_job/'export-source-native-gate.json')
        provenance = verify_original_export_native_gate(
            a.original_native_job/'native-gate.json', export_job/'export-source-native-gate.json')
        (a.output/'original-export-native-provenance-review.json').write_text(json.dumps(provenance,indent=2)+'\n')
        aliases={}
        for arm,job,gate in (('before',a.original_native_job,original),('after',a.native_job,after)):
            alias=a.output/(arm+'-full-artifact');alias.mkdir();aliases[arm]=alias
            if arm=='after':alias_runtime(job,package,alias,original['source'])
            else:
                (alias/'model-build').mkdir();g=dict(gate);g['binaries']=dict(gate['binaries'])
                for name in ('muse_glimmer_runtime_kv_probe','muse_glimmer_batching_worker'):
                    target=alias/'model-build'/name;target.symlink_to(job/'model-build'/name)
                    g['binaries'][str(target)]=sha(target)
                (alias/'native-gate.json').write_text(json.dumps(g,indent=2)+'\n')
            (alias/'full-independent-artifact').symlink_to(artifact.resolve())
            (alias/'progress.json').write_text(json.dumps(dict(status='full-artifacts-ready-for-qualification',
                controller_pid=os.getpid(),mini_gate_passed=True,all_full_artifact_hashes_verified=True,
                full_profile=profile,full_model_qualified=False,full_audit_SHA256={'independent':sha(audit_path)},
                original_and_current_source_chains_reverified=True),indent=2)+'\n')
            state('building-full-MG-matched-native-probe',native=arm)
            run(arm+'-probe','prepare_rewind_kv_probe.py',[
                '--preparation-job',alias,'--python',sys.executable,'--output',a.output/(arm+'-rewind-probe')])
            state('collecting-same-PTE-full-MG-native-latency',native=arm)
            run(arm+'-latency','run_independent_session_mini_model_latency.py',[
                '--preparation-job',alias,'--arm','independent','--full-model-profile',profile,
                '--rewind-probe-job',a.output/(arm+'-rewind-probe'),'--native-sampler','off',
                '--lock',a.lock,'--nsys',a.nsys,'--output',a.output/(arm+'-latency')])
        run('native-change-review','review_independent_session_same_artifact_native_change.py',[
            '--before-job',a.output/'before-latency','--after-job',a.output/'after-latency',
            '--output',a.output/'same-PTE-native-independent-review.json'])
        review=json.loads((a.output/'same-PTE-native-independent-review.json').read_text())
        assert review['full_model_profile']==profile and review['full_saved_FP32_tensors_equal']==9
        signals=[dict(case=cell['case'],metric=metric,**cell[metric]) for cell in review['cells']
                 for metric in ('cold_first_four_seconds','mean_stable_native_ms') if cell[metric]['positive_signal']]
        common=['--preparation-job',aliases['after'],'--full-model-profile',profile,'--lock',a.lock]
        state('collecting-full-MG-independent-capacity',positive_performance_signals=signals)
        run('capacity-control','run_independent_session_mini_capacity_control.py',common+['--output',a.output/'capacity-control'])
        run('capacity-review','review_independent_session_same_PTE_capacity.py',[
            '--job',a.output/'capacity-control','--output',a.output/'capacity-independent-review.json'])
        state('collecting-full-MG-independent-rewind',positive_performance_signals=signals)
        run('rewind','run_independent_session_same_artifact_control.py',common+[
            '--arm','independent','--rewind-probe-job',a.output/'after-rewind-probe','--output',a.output/'rewind'])
        run('rewind-review','review_independent_session_actual_rewind.py',[
            '--job',a.output/'rewind','--profile','muse-glimmer-first4','--output',a.output/'rewind-independent-review.json'])
        state('collecting-full-MG-post-shrink-graphs',positive_performance_signals=signals)
        run('graphs','run_independent_session_mini_probe.py',[
            '--probe',aliases['after']/'model-build/muse_glimmer_runtime_kv_probe','--native-gate',aliases['after']/'native-gate.json',
            '--artifact',artifact,'--full-model-profile',profile,'--arm','independent','--automatic-reclaim',
            '--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
            '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
            '--job-dir',a.output/'graphs','--lock',a.lock,'--nsys',a.nsys])
        trace=a.output/'graphs/round-1/cuda-graphs.nsys-rep';sqlite=trace.with_suffix('.sqlite');assert not sqlite.exists()
        expected=sha(trace)
        with (a.output/'graph-export.log').open('x') as log:
            subprocess.run([str(a.nsys),'export','--type=sqlite','--output='+str(sqlite),str(trace)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        assert sha(trace)==expected
        run('graph-review','review_independent_session_mini_graph_trace.py',[
            '--runtime-job',a.output/'graphs','--sqlite',sqlite,'--output',a.output/'graph-independent-review.json'])
        capacity=json.loads((a.output/'capacity-independent-review.json').read_text())
        rewind=json.loads((a.output/'rewind-independent-review.json').read_text())
        graph=json.loads((a.output/'graph-independent-review.json').read_text())
        assert capacity['bitwise_allocation_equivalence_verified']
        assert len(rewind['checks'])==48 and rewind['invalid_out_of_context_rewinds_rejected']==24
        assert graph['all_seven_static_methods_after_shrink_verified'] and graph['automatic_reclaim_verified_without_compact_calls']
        (a.output/'result.json').write_text(json.dumps(dict(full_profile=profile,
            full_independent_storage_and_observed_numerics_pass=True,
            fixed_token_same_PTE_runtime_performance_pass=not signals,positive_performance_signals=signals,
            full_model_qualified=False,main_candidate_accepted=False,
            raw_reviews_SHA256={name:sha(a.output/name) for name in ('same-PTE-native-independent-review.json','capacity-independent-review.json','rewind-independent-review.json','graph-independent-review.json')},
            global_layout_comparison_not_collected=True,HTTP_F1_and_main_pending=True),indent=2)+'\n')
        state('full-MG-single-artifact-native-controls-collected',positive_performance_signals=signals)
    except Exception as e:
        state('failed',error=str(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
