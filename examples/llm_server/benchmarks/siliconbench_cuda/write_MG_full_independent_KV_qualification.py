"""Qualify the full MG runtime from complete, re-reviewed raw controls and F1.

Original consecutive signals remain evidence. A matched four-round ABBA
control must resolve them on the same compiled weights, inputs and GPU.
This receipt permits main collection; it never accepts a main result.
"""
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from independent_session_full_native_binding import sha, verify_runtime
from independent_session_full_http_binding import verified_mg_full_http


def require_complete_control_states(jobs):
    for job, terminal in jobs:
        assert json.loads((Path(job)/'progress.json').read_text())['status'] == terminal, job


def current_GPU_UUID():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    assert len(rows) == 1 and rows[0].startswith('GPU-')
    return rows[0].strip()


def verify_MG_full_qualification(path, args, binding):
    path = Path(path)
    q = json.loads(path.read_text())
    package = Path(args.package)
    assert q['producer_source_SHA256'] == sha(package/Path(__file__).name)
    assert q['full_profile'] == args.full_model_profile == 'muse-glimmer-52layers'
    assert q['main_candidate_accepted'] is False
    assert q['full_model_qualified'] and not q['unresolved_correctness_or_performance_signals']
    for key in ('full_model_correctness_pass', 'full_task_F1_independently_reviewed',
                'full_fixed_token_performance_pass', 'full_storage_rewind_graph_pass', 'full_HTTP_memory_pass'):
        assert q[key] is True, key
    runtime_hash = sha(args.native_job/'native-gate.json')
    assert q['native_runtime_gate_SHA256'] == runtime_hash
    assert q['HTTP_parent_result_SHA256'] == sha(args.HTTP_job/'result.json')
    assert q['native_lifecycle_result_SHA256'] == sha(args.native_lifecycle_job/'result.json')
    artifact = binding['artifacts']['independent']
    assert q['artifact_SHA256'] == {name:sha(artifact/name) for name in ('model.pte', 'aoti_cuda_blob.ptd')}
    for name, expected in q['evidence_files_SHA256'].items():
        assert sha(Path(name)) == expected, name
    quality = json.loads(Path(q['raw_quality_review']).read_text())
    assert quality['native_runtime_gate_SHA256'] == runtime_hash
    assert quality['artifact_SHA256'] == q['artifact_SHA256']
    assert quality['full_profile'] == q['full_profile']
    assert quality['complete'] and quality['reviewed_shots'] == 20 and quality['actual_response_count'] == 9936
    assert quality['raw_input_request_and_runtime_binding_checked'] and quality['full_task_raw_quality_pass']
    assert quality['full_task_F1_independently_reviewed']
    assert len(quality['shots']) == 20
    expected = {(prefix, c, shot) for prefix in ('off','on') for c in (1,2,4,8,16) for shot in (0,5)}
    assert {(r['prefix'],r['concurrency'],r['shots']) for r in quality['shots']} == expected
    for row in quality['shots']:
        assert row['count'] == (1146 if row['concurrency'] in (1,16) else 64)
        assert row['errors'] == row['parse_failures'] == row['final_content_scoring_disagreements'] == 0
        assert row['finish_reasons'] == {'stop':row['count']}
    resolution = q['performance_resolution']
    original = json.loads((args.HTTP_job/'result.json').read_text())
    assert resolution['original_consecutive_signals'] == original['positive_performance_signals']
    assert resolution['original_gates_unchanged'] and resolution['no_automatic_cold_or_recapture_waiver']
    matched = json.loads(Path(resolution['matched_result']).read_text())
    control = json.loads(Path(resolution['matched_control_result']).read_text())
    reviewed = json.loads(Path(resolution['recomputed_native_review']).read_text())
    assert matched['full_profile'] == control['full_model_profile'] == reviewed['full_model_profile'] == q['full_profile']
    assert matched['matched_fixed_token_performance_pass'] and not matched['positive_performance_signals']
    assert matched['original_gates_unchanged'] and matched['original_consecutive_signals'] == resolution['original_consecutive_signals']
    assert control['four_rounds_discard_first'] and control['ABBA_native_order']
    assert control['initial_KV_capacity'] == 512 and control['collected_arms'] == ['independent']
    assert not control['positive_performance_signals'] and len(control['actual_child_results']) == 8
    clocks = Path(resolution['matched_control_result']).parent/'GPU-clocks-power.jsonl'
    assert sha(clocks) == control['clock_power_samples_SHA256']
    measured_UUIDs = {json.loads(line)['actual_GPU_query'].split(',')[0].strip() for line in clocks.read_text().splitlines()}
    assert measured_UUIDs == {q['qualification_GPU_UUID']} == {current_GPU_UUID()}
    assert reviewed['full_saved_FP32_tensors_equal'] == 9 and reviewed['selected_argmax_values_equal'] == 3376
    assert reviewed['native_change_preserves_observed_model_numerics']
    assert len(reviewed['cells']) == 9 and len(reviewed['records']) == 4
    for cell in reviewed['cells']:
        assert cell['arm'] == 'independent'
        for metric in ('cold_first_four_seconds','mean_stable_native_ms'):
            assert not cell[metric]['positive_signal'], (cell['case'],metric)
    for result in control['actual_child_results']:
        assert sha(Path(result['result'])) == result['SHA256']
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package','native-job','native-lifecycle-job','HTTP-job','matched-native-job',
                 'mini-qualification','quality-job','public-inputs','memory-worker-proof',
                 'http-python-dependencies-job','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Preserve every prior qualification attempt'
    # Reject incomplete/failed controls before creating a qualification directory.
    require_complete_control_states(((a.native_lifecycle_job,'full-MG-single-artifact-native-controls-collected'),
                          (a.HTTP_job,'full-HTTP-memory-collected-pending-F1-and-main'),
                          (a.matched_native_job,'full-MG-matched-native-diagnostic-collected'),
                          (a.quality_job,'collected-pending-independent-raw-review')))
    native = verify_runtime(a.native_job,a.package)
    a.full_model_profile = 'muse-glimmer-52layers'
    a.mini_profile = 'muse-glimmer-first4';a.arm = 'independent'
    a.full_native_job = a.native_job;a.full_MG_native_lifecycle_job = a.native_lifecycle_job
    a.full_MG_mini_qualification = a.mini_qualification
    a.preparation_job = a.native_lifecycle_job/'after-full-artifact'
    a.preserved_global_HTTP_failure_job = None
    binding = verified_mg_full_http(a,a.package)
    matched = json.loads((a.matched_native_job/'result.json').read_text())
    assert matched['matched_fixed_token_performance_pass'] and not matched['positive_performance_signals']
    assert matched['original_gates_unchanged'] and not matched['full_model_qualified']
    for name, expected in matched['original_evidence_SHA256'].items():assert sha(Path(name)) == expected,name
    a.output.mkdir(parents=True)
    def run(label, script, argv):
        command = [sys.executable,'-B',str(a.package/script),*map(str,argv)]
        (a.output/(label+'-command.json')).write_text(json.dumps({'argv':command},indent=2)+'\n')
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=3600)
    review = a.output/'matched-native-raw-review.json'
    control = a.matched_native_job/'matched-control'
    run('matched-native-review','review_independent_session_same_artifact_native_change.py',[
        '--before-job',control/'before','--after-job',control/'after','--output',review])
    recomputed = json.loads(review.read_text())
    actual = json.loads((control/'same-PTE-native-independent-review.json').read_text())
    assert recomputed['cells'] == actual['cells'] and recomputed['records'] == actual['records']
    for cell in recomputed['cells']:
        for metric in ('cold_first_four_seconds','mean_stable_native_ms'):
            assert not cell[metric]['positive_signal'], (cell['case'],metric)
    quality = a.output/'full-task-raw-review.json'
    run('quality-review','review_independent_full_task_quality.py',[
        '--quality-job',a.quality_job,'--public-inputs',a.public_inputs,'--native-job',a.native_job,
        '--full-model-profile',a.full_model_profile,'--output',quality])
    q = json.loads(quality.read_text())
    assert q['complete'] and q['full_task_raw_quality_pass'] and q['actual_response_count'] == 9936
    evidence = dict(q['evidence_files_SHA256'])
    suffixes = ('.json','.jsonl','.f32','.log','.sqlite','.nsys-rep','.xml')
    HTTP = json.loads((a.HTTP_job/'result.json').read_text())
    HTTP_raw = Path(HTTP.get('completed_HTTP_job',str(a.HTTP_job/'HTTP-memory')))
    assert (HTTP_raw/'result.json').is_file()
    for root in (a.native_lifecycle_job,a.HTTP_job,HTTP_raw,control,a.quality_job):
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in suffixes:continue
            if any(part in ('executorch','model-build','build','__pycache__') for part in path.relative_to(root).parts):continue
            # Artifact directories are symlinks and are never traversed here.
            assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),path
            evidence[str(path.resolve())] = sha(path)
    for path in (review,quality,a.matched_native_job/'result.json',a.native_job/'native-gate.json'):
        evidence[str(path.resolve())] = sha(path)
    artifact = binding['artifacts']['independent']
    result = dict(full_profile=a.full_model_profile,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        qualification_GPU_UUID=current_GPU_UUID(),
        producer_source_SHA256=sha(Path(__file__)),native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),
        artifact_SHA256={name:sha(artifact/name) for name in ('model.pte','aoti_cuda_blob.ptd')},
        HTTP_parent_result_SHA256=sha(a.HTTP_job/'result.json'),
        native_lifecycle_result_SHA256=sha(a.native_lifecycle_job/'result.json'),
        full_model_correctness_pass=True,full_task_F1_independently_reviewed=True,
        full_fixed_token_performance_pass=True,full_storage_rewind_graph_pass=True,full_HTTP_memory_pass=True,
        unresolved_correctness_or_performance_signals=[],raw_quality_review=str(quality.resolve()),
        performance_resolution=dict(original_consecutive_signals=HTTP['positive_performance_signals'],
            original_gates_unchanged=True,no_automatic_cold_or_recapture_waiver=True,
            matched_result=str((a.matched_native_job/'result.json').resolve()),
            matched_control_result=str((control/'result.json').resolve()),recomputed_native_review=str(review.resolve()),
            rule='Same-PTE four-round ABBA, discard first; no cold or stable change above combined3SE. Original signals remain preserved.'),
        evidence_files_SHA256=evidence,full_model_qualified=True,main_candidate_accepted=False,
        F1_accuracy_is_not_universal_task_correctness=True,
        main_performance_and_same_GPU_accepted_baseline_comparison_pending=True)
    pending = a.output/'qualification.pending.json'
    pending.write_text(json.dumps(result,indent=2)+'\n')
    verify_MG_full_qualification(pending,a,binding)
    assert verify_runtime(a.native_job,a.package) == native
    pending.rename(a.output/'full-qualification.json')
    print(json.dumps({'full_model_qualified':True,'main_candidate_accepted':False,
        'qualification':str(a.output/'full-qualification.json'),'actual_F1_responses':9936}))


if __name__ == '__main__':main()
