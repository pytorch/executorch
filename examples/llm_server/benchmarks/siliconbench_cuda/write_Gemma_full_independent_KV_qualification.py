"""Qualify full Gemma from raw native, storage, HTTP and complete task evidence.

The same-PTE matched baseline and prior accepted main are separate. This
receipt allows collection only; final main and baseline comparison stay pending.
"""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime,verify_qualification
from independent_session_full_http_binding import verified_gemma_full_http
from write_MG_full_independent_KV_qualification import current_GPU_UUID,require_complete_control_states


def verify_Gemma_full_qualification(path,args,binding):
 q=json.loads(Path(path).read_text());package=Path(args.package)
 assert q['producer_source_SHA256']==sha(package/Path(__file__).name)
 assert q['full_profile']==args.full_model_profile=='gemma4-48layers'
 assert q['full_model_qualified'] and not q['main_candidate_accepted']
 assert not q['unresolved_correctness_or_performance_signals']
 for key in ('full_model_correctness_pass','full_task_F1_independently_reviewed','full_fixed_token_performance_pass','full_storage_rewind_graph_pass','full_HTTP_memory_pass'):assert q[key]
 native_hash=sha(args.native_job/'native-gate.json');assert q['native_runtime_gate_SHA256']==native_hash
 control_root=Path(q['full_controls_job']);quality_root=Path(q['quality_follower_job'])
 assert args.fixed_job.resolve()==(control_root/'full-fixed').resolve()
 assert args.HTTP_job.resolve()==(control_root/'full-HTTP').resolve()
 require_complete_control_states(((control_root,'full48-allocator-controls-collected-pending-F1-and-main'),(quality_root,'fresh-full-Gemma-native-ABBA-and-quality-collected')))
 assert q['full_controls_result_SHA256']==sha(control_root/'result.json')
 assert q['quality_follower_result_SHA256']==sha(quality_root/'result.json')
 verify_qualification(Path(q['mini_qualification']),'gemma4-first6','7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a',args.native_job)
 assert q['mini_qualification_SHA256']==sha(Path(q['mini_qualification']))
 artifact=binding['artifacts']['independent'];assert q['artifact_SHA256']=={n:sha(artifact/n) for n in ('model.pte','aoti_cuda_blob.ptd')}
 for name,expected in q['evidence_files_SHA256'].items():assert sha(Path(name))==expected,name
 quality=json.loads(Path(q['raw_quality_review']).read_text())
 assert quality['native_runtime_gate_SHA256']==native_hash and quality['artifact_SHA256']==q['artifact_SHA256']
 assert quality['full_profile']=='gemma4-48layers' and quality['complete']
 assert quality['reviewed_shots']==20 and quality['actual_response_count']==9936
 assert quality['raw_input_request_and_runtime_binding_checked'] and quality['full_task_raw_quality_pass'] and quality['full_task_F1_independently_reviewed']
 expected={(prefix,c,shot) for prefix in ('off','on') for c in (1,2,4,8,16) for shot in (0,5)}
 assert len(quality['shots'])==20 and {(r['prefix'],r['concurrency'],r['shots']) for r in quality['shots']}==expected
 for row in quality['shots']:
  assert row['count']==(1146 if row['concurrency'] in (1,16) else 64)
  assert row['errors']==row['parse_failures']==row['final_content_scoring_disagreements']==0
  assert row['finish_reasons']=={'stop':row['count']}
 matched=quality_root/'matched-control';control=json.loads((matched/'result.json').read_text())
 assert control['full_model_profile']=='gemma4-48layers' and control['collected_arms']==['independent']
 assert control['four_rounds_discard_first'] and control['ABBA_native_order'] and control['initial_KV_capacity']==512
 assert not control['positive_performance_signals'] and len(control['actual_child_results'])==8
 assert sha(matched/'GPU-clocks-power.jsonl')==control['clock_power_samples_SHA256']
 UUIDs={json.loads(line)['actual_GPU_query'].split(',')[0].strip() for line in (matched/'GPU-clocks-power.jsonl').read_text().splitlines()}
 assert UUIDs=={q['qualification_GPU_UUID']}=={current_GPU_UUID()}
 for child in control['actual_child_results']:assert sha(Path(child['result']))==child['SHA256']
 reviewed=json.loads(Path(q['matched_native_raw_review']).read_text())
 assert reviewed['full_model_profile']=='gemma4-48layers' and reviewed['native_change_preserves_observed_model_numerics']
 assert reviewed['full_saved_FP32_tensors_equal']==9 and reviewed['selected_argmax_values_equal']==3672
 assert len(reviewed['cells'])==9 and len(reviewed['records'])==4
 for c in reviewed['cells']:
  assert c['arm']=='independent'
  for key in ('cold_first_four_seconds','mean_stable_native_ms'):assert not c[key]['positive_signal'],(c['case'],key)
 original=json.loads((control_root/'full-HTTP/result.json').read_text())
 assert q['performance_resolution']['original_consecutive_signals']==original['positive_performance_signals']
 assert q['performance_resolution']['original_gates_unchanged'] and q['performance_resolution']['no_automatic_cold_or_recapture_waiver']
 lifecycle=control_root/'full-lifecycle';doc=json.loads((lifecycle/'result.json').read_text())
 assert doc['full_profile']=='gemma4-48layers' and doc['full_storage_rewind_graph_pass']
 assert doc['native_gate_SHA256']==native_hash
 for name,expected in doc['raw_reviews_SHA256'].items():assert sha(lifecycle/name)==expected
 for name in ('capacity','rewind','graph'):
  assert sha(Path(q['recomputed_storage_reviews'][name]))==q['recomputed_storage_reviews_SHA256'][name]
 capacity=json.loads(Path(q['recomputed_storage_reviews']['capacity']).read_text())
 rewind=json.loads(Path(q['recomputed_storage_reviews']['rewind']).read_text())
 graph=json.loads(Path(q['recomputed_storage_reviews']['graph']).read_text())
 assert capacity['bitwise_allocation_equivalence_verified'] and len(capacity['fresh_native_PIDs'])==8
 assert rewind['raw_logits_protocols_and_actual_rewind_verified'] and len(rewind['checks'])==48
 assert rewind['invalid_out_of_context_rewinds_rejected']==24 and rewind['collected_arms']==['independent']
 assert graph['actual_CUDA_trace'] and graph['all_seven_static_methods_after_shrink_verified'] and graph['automatic_reclaim_verified_without_compact_calls']
 http=json.loads(Path(q['recomputed_HTTP_review']).read_text())
 assert http['full_model_profile']=='gemma4-48layers' and http['native_allocator_memory_verified'] and http['automatic_OFF_c16_to_c1_reclaim_verified']
 assert http['collected_arms']==['independent'] and http['actual_requests']==256 and http['server_runs']==8
 return q


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('package','full-controls-job','quality-follower-job','native-job','mini-qualification','memory-worker-proof','Gemma-source-preparation-job','http-python-dependencies-job','tokenizer','public-inputs','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists()
 require_complete_control_states(((a.full_controls_job,'full48-allocator-controls-collected-pending-F1-and-main'),(a.quality_follower_job,'fresh-full-Gemma-native-ABBA-and-quality-collected')))
 native=verify_runtime(a.native_job,a.package);a.full_model_profile='gemma4-48layers';a.mini_profile='gemma4-first6';a.full_native_job=a.native_job;a.arm='independent';a.preserved_global_HTTP_failure_job=None
 a.fixed_job=a.full_controls_job/'full-fixed';a.HTTP_job=a.full_controls_job/'full-HTTP';a.preparation_job=a.fixed_job/'full-pair'
 binding=verified_gemma_full_http(a,a.package)
 verify_qualification(a.mini_qualification,a.mini_profile,'7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a',a.native_job)
 parent=json.loads((a.quality_follower_job/'result.json').read_text());assert parent['raw_full_quality_pass'] and not parent['positive_performance_signals']
 assert parent['original_evidence_preserved'] and not parent['full_model_qualified'] and not parent['main_candidate_accepted']
 a.output.mkdir(parents=True)
 def run(label,script,argv):
  cmd=[sys.executable,'-B',str(a.package/script),*map(str,argv)]
  with (a.output/(label+'.log')).open('x') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=3600)
 matched=a.quality_follower_job/'matched-control';review=a.output/'matched-native-raw-review.json'
 run('matched-native','review_independent_session_same_artifact_native_change.py',['--before-job',matched/'before','--after-job',matched/'after','--output',review])
 observed=json.loads((matched/'same-PTE-native-independent-review.json').read_text());actual=json.loads(review.read_text());assert actual['cells']==observed['cells'] and actual['records']==observed['records']
 quality=a.output/'full-task-raw-review.json'
 run('quality','review_independent_full_task_quality.py',['--quality-job',a.quality_follower_job/'full-quality','--public-inputs',a.public_inputs,'--native-job',a.native_job,'--full-model-profile','gemma4-48layers','--output',quality])
 lifecycle=a.full_controls_job/'full-lifecycle';reviews={}
 for name,script,argv in (
  ('capacity','review_independent_session_same_PTE_capacity.py',['--job',lifecycle/'capacity-control']),
  ('rewind','review_independent_session_actual_rewind.py',['--job',lifecycle/'same-artifact-rewind','--profile','gemma4-first6']),
  ('graph','review_independent_session_mini_graph_trace.py',['--runtime-job',lifecycle/'graph-lifecycle','--sqlite',lifecycle/'graph-lifecycle/round-1/cuda-graphs.sqlite'])):
  dest=a.output/(name+'-raw-review.json');run(name,script,argv+['--output',dest]);reviews[name]=str(dest.resolve())
 http=a.output/'HTTP-raw-review.json'
 run('HTTP','review_independent_session_http_lifecycle.py',['--job',a.HTTP_job/'HTTP-memory','--require-native-memory','--output',http])
 evidence=dict(json.loads(quality.read_text())['evidence_files_SHA256'])
 for root in (a.full_controls_job,a.quality_follower_job,a.output):
  for path in root.rglob('*'):
   if not path.is_file() or path.is_symlink() or path.suffix not in ('.json','.jsonl','.f32','.log','.sqlite','.nsys-rep','.xml'):continue
   if any(part in ('executorch','model-build','build','__pycache__') for part in path.relative_to(root).parts):continue
   evidence[str(path.resolve())]=sha(path)
 evidence[str(a.mini_qualification.resolve())]=sha(a.mini_qualification)
 artifact=binding['artifacts']['independent'];original=json.loads((a.HTTP_job/'result.json').read_text())
 q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_profile='gemma4-48layers',producer_source_SHA256=sha(Path(__file__)),qualification_GPU_UUID=current_GPU_UUID(),native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),artifact_SHA256={n:sha(artifact/n) for n in ('model.pte','aoti_cuda_blob.ptd')},full_controls_job=str(a.full_controls_job.resolve()),quality_follower_job=str(a.quality_follower_job.resolve()),full_controls_result_SHA256=sha(a.full_controls_job/'result.json'),quality_follower_result_SHA256=sha(a.quality_follower_job/'result.json'),mini_qualification=str(a.mini_qualification.resolve()),mini_qualification_SHA256=sha(a.mini_qualification),full_model_correctness_pass=True,full_task_F1_independently_reviewed=True,full_fixed_token_performance_pass=True,full_storage_rewind_graph_pass=True,full_HTTP_memory_pass=True,unresolved_correctness_or_performance_signals=[],raw_quality_review=str(quality.resolve()),matched_native_raw_review=str(review.resolve()),recomputed_storage_reviews=reviews,recomputed_storage_reviews_SHA256={n:sha(Path(v)) for n,v in reviews.items()},recomputed_HTTP_review=str(http.resolve()),performance_resolution=dict(original_consecutive_signals=original['positive_performance_signals'],original_gates_unchanged=True,no_automatic_cold_or_recapture_waiver=True,same_PTE_ABBA_baseline_is_separate_from_prior_accepted_main=True),evidence_files_SHA256=evidence,full_model_qualified=True,main_candidate_accepted=False,main_performance_and_same_GPU_accepted_baseline_comparison_pending=True)
 pending=a.output/'qualification.pending.json';pending.write_text(json.dumps(q,indent=2)+'\n');verify_Gemma_full_qualification(pending,a,binding)
 assert verify_runtime(a.native_job,a.package)==native;pending.rename(a.output/'full-qualification.json')
 print(json.dumps(dict(full_model_qualified=True,main_candidate_accepted=False,qualification=str(a.output/'full-qualification.json'))))

if __name__=='__main__':main()
