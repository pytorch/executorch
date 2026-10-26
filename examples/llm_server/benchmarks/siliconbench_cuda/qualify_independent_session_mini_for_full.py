"""Recheck task-owned raw mini evidence before permitting full export preparation.

This gate qualifies storage and measured mini performance only. Separately
compiled logits and full-model F1/performance remain pending; no main result is
accepted or overwritten. Reviews read actual protocols, saved FP32 tensors and
CUDA trace events. Positive cold/steady timing signals keep the export gate shut.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('mini-job','native-job','previous-latency-job','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--profile',choices=['muse-glimmer-first4','gemma4-first6'],required=True)
 p.add_argument('--Gemma-source-preparation-job',type=Path)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 current=load(a.mini_job/'progress.json')
 assert current['status']=='growth-allocator-mini-controls-collected-pending-independent-review',current
 collected=load(a.mini_job/'result.json');g=load(a.native_job/'native-gate.json')
 assert g['native_GPU_tests_pass'] and g['test_count']==74
 assert collected['native_gate_SHA256']==sha(a.native_job/'native-gate.json')
 assert g['native_tests_xml_SHA256']==sha(a.native_job/'native-tests.xml')
 tests=list(ET.parse(a.native_job/'native-tests.xml').getroot().iter('testcase'))
 assert len(tests)==74 and not any(t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None for t in tests)
 for path,expected in g['binaries'].items():assert sha(path)==expected,path
 for name,r in collected['jobs'].items():assert sha(r['result'])==r['SHA256'],name
 def review(name,script,args):
  dest=a.output/(name+'.json')
  with (a.output/(name+'.log')).open('x') as log:
   subprocess.run([sys.executable,'-B',str(package/script),*map(str,args),'--output',str(dest)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  return load(dest)
 gemma=a.profile=='gemma4-first6'
 latency=review('fixed-latency','review_independent_session_gemma_fixed_latency.py' if gemma else 'review_independent_session_mg_fixed_latency.py',['--job',a.mini_job/'model-latency'])
 capacity=review('same-PTE-capacity','review_independent_session_same_PTE_capacity.py',['--job',a.mini_job/'capacity-control'])
 rewind=review('actual-rewind','review_independent_session_actual_rewind.py',['--job',a.mini_job/'same-artifact-rewind','--profile',a.profile])
 graph=review('actual-graph','review_independent_session_mini_graph_trace.py',['--runtime-job',a.mini_job/'graph-lifecycle','--sqlite',a.mini_job/'graph-lifecycle/round-1/cuda-graphs.sqlite'])
 http=review('HTTP-memory','review_independent_session_http_lifecycle.py',['--job',a.mini_job/'HTTP-memory','--require-native-memory'])
 native=review('same-artifact-native-change','review_independent_session_same_artifact_native_change.py',['--before-job',a.previous_latency_job,'--after-job',a.mini_job/'model-latency'])
 assert capacity['bitwise_allocation_equivalence_verified'] and len(capacity['fresh_native_PIDs'])==8
 assert rewind['raw_logits_protocols_and_actual_rewind_verified'] and len(rewind['checks'])==96 and rewind['invalid_out_of_context_rewinds_rejected']==48
 assert graph['actual_CUDA_trace'] and graph['all_seven_static_methods_after_shrink_verified'] and graph['automatic_reclaim_verified_without_compact_calls']
 assert http['raw_streams_usage_timestamps_and_hashes_verified'] and http['matched_requests']==256 and http['native_allocator_memory_verified'] and http['automatic_OFF_c16_to_c1_reclaim_verified']
 assert native['native_change_preserves_observed_model_numerics'] and native['full_saved_FP32_tensors_equal']==18
 signals=[]
 for c in latency['aggregates']:
  for key in ('cold_first_four_seconds','mean_stable_native_ms'):
   r=c[key];delta=r['arms']['independent']['mean']-r['arms']['global']['mean']
   if delta>r['combined_3SE']:signals.append({'comparison':'global-to-independent','case':c['case'],'metric':key,'delta_percent':r['delta_percent'],'delta':delta,'combined_3SE':r['combined_3SE']})
 for c in native['cells']:
  if c['arm']!='independent':continue
  for key in ('cold_first_four_seconds','mean_stable_native_ms'):
   r=c[key]
   if r['positive_signal']:signals.append({'comparison':'same-PTE-native-change','case':c['case'],'metric':key,'delta_percent':r['delta_percent'],'delta':r['delta'],'combined_3SE':r['combined_3SE']})
 artifact=a.native_job/'same-artifact-mini-preparation/mini-independent-artifact'
 audit=load(artifact/'independent-mini-artifact-audit.json')
 expected='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a' if gemma else '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
 assert audit['GGUF_sha256']==expected and audit['independent_capacity']
 for name,r in audit['artifacts'].items():assert sha(artifact/name)==r['sha256'],name
 evidence={}
 for directory in (a.mini_job,a.output):
  for path in directory.rglob('*'):
   if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.f32','.sqlite','.nsys-rep') and not any(part in ('model-build','executorch','build') for part in path.relative_to(directory).parts):
    evidence[str(path.resolve())]=sha(path)
 for path in (a.native_job/'native-gate.json',a.native_job/'native-tests.xml',artifact/'independent-mini-artifact-audit.json'):
  evidence[str(path.resolve())]=sha(path)
 receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mini_profile':a.profile,
  'mini_storage_correctness_pass':True,'mini_cold_and_steady_performance_pass':not signals,
  'mini_HTTP_memory_pass':True,'full_model_export_for_qualification_allowed':not signals,
  'full_model_qualified':False,'main_candidate_accepted':False,'GGUF_SHA256':expected,
  'runtime_native_gate_SHA256':sha(a.native_job/'native-gate.json'),'evidence_files_SHA256':evidence,
  'positive_performance_signals':signals,'HTTP_natural_lengths_equal':http['completion_lengths_equal_between_arms'],
  'HTTP_TPS_or_F1_acceptance':False,'cross_export_strict_numerical_equivalence':False,
  'scope':'Mini storage/actual graph/memory and fixed-token cold/steady gate only. Full-model correctness, F1, same-GPU performance and sampled memory remain required. Raw timings discard round1 and use rounds2–4;3SE is a noise heuristic, not a confidence guarantee.'}
 if gemma:
  assert a.Gemma_source_preparation_job
  prep=a.Gemma_source_preparation_job/'source-preparation.json'
  assert sha(prep)==audit['Gemma_source_preparation_SHA256']
  receipt['Gemma_source_preparation_SHA256']=sha(prep);receipt['evidence_files_SHA256'][str(prep.resolve())]=sha(prep)
 (a.output/'mini-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:receipt[k] for k in ('mini_profile','mini_storage_correctness_pass','mini_cold_and_steady_performance_pass','full_model_export_for_qualification_allowed','positive_performance_signals')}))

if __name__=='__main__':main()
