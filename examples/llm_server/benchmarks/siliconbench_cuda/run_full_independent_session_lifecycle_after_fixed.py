"""Run full52/48-layer storage/rewind/graph gates after fixed-token collection.

Preserve both compiled full artifacts and all prior runs. A timing signal does
not disappear when storage checks pass; no main result is accepted here.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import verify_runtime,sha
from independent_session_mini_profiles import FULL_PROFILES


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('fixed-job','native-job','output','lock','nsys'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--full-model-profile',choices=FULL_PROFILES,required=True)
 p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
 p.add_argument('--preserved-global-rewind-failure-job',type=Path,
  help='Actual prior global-only executor failure; retain it and collect unchanged independent stress')
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=a.package.resolve()
 mini_profile='gemma4-first6' if a.full_model_profile=='gemma4-48layers' else 'muse-glimmer-first4'
 def state(status,**extra):
  tmp=a.output/'progress.pending.json'
  tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_model_qualified=False,
   main_candidate_accepted=False,**extra),indent=2)+'\n');tmp.replace(a.output/'progress.json')
 def run(label,script,argv):
  command=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:
   subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=21600)
 try:
  deadline=time.monotonic()+10*3600
  while True:
   d=json.loads((a.fixed_job/'progress.json').read_text())
   if d['status']=='full-fixed-token-collected-pending-other-full-gates':break
   assert d['status']!='failed',d
   proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
   assert proc.is_file() and any(name in proc.read_bytes() for name in (
    b'run_gemma_full_fixed_qualification_after_exports.py',b'run_full_independent_session_fixed_qualification.py'))
   assert time.monotonic()<deadline,'Verified-live observation expired; preserve existing jobs'
   state('waiting-confirmed-live-full-fixed-control',PID=d['controller_pid'],fixed_status=d['status'])
   time.sleep(20)
  gate=verify_runtime(a.native_job,package);pair=a.fixed_job/'full-pair'
  prep=json.loads((pair/'progress.json').read_text())
  assert prep['mini_gate_passed'] and prep['all_full_artifact_hashes_verified']
  fixed=json.loads((a.fixed_job/'result.json').read_text());assert fixed['full_profile']==a.full_model_profile
  common=['--preparation-job',pair,'--mini-profile',mini_profile,'--full-model-profile',a.full_model_profile,'--lock',a.lock]
  state('collecting-full-same-PTE-capacity-and-auto-reclaim')
  run('capacity-control','run_independent_session_mini_capacity_control.py',common+['--output',a.output/'capacity-control'])
  run('capacity-review','review_independent_session_same_PTE_capacity.py',[
   '--job',a.output/'capacity-control','--output',a.output/'capacity-independent-review.json'])
  preserved_global_failure=None
  rewind_arms=['global','independent']
  if a.preserved_global_rewind_failure_job:
   failed_root=a.preserved_global_rewind_failure_job
   failure_path=failed_root/'progress.json';failure=json.loads(failure_path.read_text())
   assert failure['status']=='failed' and 'actual executor step failed' in failure['error']
   handle=Path('/proc')/str(failure['controller_pid'])/'cmdline';assert not handle.exists() or b'run_independent_session_same_artifact_control.py' not in handle.read_bytes()
   failed_command=json.loads((failed_root.parent/'same-artifact-rewind-command.json').read_text())['argv']
   original_pair=Path(failed_command[failed_command.index('--preparation-job')+1])
   old_audit=original_pair/'full-global-artifact/independent-full-artifact-audit.json'
   new_audit=pair/'full-global-artifact/independent-full-artifact-audit.json'
   assert json.loads(old_audit.read_text())['artifacts']==json.loads(new_audit.read_text())['artifacts']
   protocol=failed_root/'round-1/global/protocol.jsonl'
   entries=[json.loads(l) for l in protocol.read_text().splitlines()]
   assert entries[-1]['response'].get('failed') and entries[-2]['request']['op']=='step'
   assert entries[-1]['response']['detail']=='actual executor step failed; do not continue owners'
   preserved_global_failure={'path':str(failed_root.resolve()),'progress_SHA256':sha(failure_path),
    'protocol_SHA256':sha(protocol),'prior_command_SHA256':sha(failed_root.parent/'same-artifact-rewind-command.json'),
    'same_global_PTE_PTD':True,'actual_executor_failure_preserved':True,'cause_not_yet_proven':True,
    'failed_global_test_is_not_a_pass':True}
   rewind_arms=['independent']
  state('collecting-full-actual-rewind')
  run('same-artifact-rewind','run_independent_session_same_artifact_control.py',common+[
   '--output',a.output/'same-artifact-rewind','--rewind-probe-job',a.fixed_job/'rewind-probe']+(['--arm','independent'] if preserved_global_failure else []))
  run('rewind-review','review_independent_session_actual_rewind.py',[
   '--job',a.output/'same-artifact-rewind','--profile',mini_profile,'--output',a.output/'rewind-independent-review.json'])
  artifact=pair/'full-independent-artifact';audit=json.loads((artifact/'independent-full-artifact-audit.json').read_text())
  state('collecting-full-post-growth-actual-CUDA-graphs')
  run('graph-lifecycle','run_independent_session_mini_probe.py',[
   '--probe',pair/'model-build/muse_glimmer_runtime_kv_probe','--artifact',artifact,
   '--native-gate',pair/'native-gate.json','--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
   '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
   '--arm','independent','--mini-profile',mini_profile,'--full-model-profile',a.full_model_profile,
   '--automatic-reclaim','--job-dir',a.output/'graph-lifecycle','--lock',a.lock,'--nsys',a.nsys])
  trace=a.output/'graph-lifecycle/round-1/cuda-graphs.nsys-rep';trace_sha=sha(trace)
  sqlite=a.output/'graph-lifecycle/round-1/cuda-graphs.sqlite';assert not sqlite.exists()
  with (a.output/'graph-sqlite-export.log').open('x') as log:
   subprocess.run([str(a.nsys),'export','--type=sqlite','--output='+str(sqlite),str(trace)],
    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  assert sha(trace)==trace_sha
  run('graph-review','review_independent_session_mini_graph_trace.py',[
   '--runtime-job',a.output/'graph-lifecycle','--sqlite',sqlite,'--output',a.output/'graph-independent-review.json'])
  capacity=json.loads((a.output/'capacity-independent-review.json').read_text())
  rewind=json.loads((a.output/'rewind-independent-review.json').read_text())
  graph=json.loads((a.output/'graph-independent-review.json').read_text())
  assert capacity['bitwise_allocation_equivalence_verified'] and len(capacity['fresh_native_PIDs'])==8
  assert rewind['raw_logits_protocols_and_actual_rewind_verified'] and len(rewind['checks'])==48*len(rewind_arms)
  assert rewind['invalid_out_of_context_rewinds_rejected']==24*len(rewind_arms)
  assert graph['actual_CUDA_trace'] and graph['all_seven_static_methods_after_shrink_verified']
  assert graph['automatic_reclaim_verified_without_compact_calls']
  assert rewind['collected_arms']==rewind_arms
  result=dict(full_profile=a.full_model_profile,full_storage_rewind_graph_pass=True,
   candidate_independent_rewind_pass=True,rewind_test_arms=rewind_arms,
   preserved_global_rewind_failure=preserved_global_failure,global_rewind_pass=not bool(preserved_global_failure),
   fixed_token_performance_pass=fixed['fixed_token_performance_pass'],positive_performance_signals=fixed['positive_performance_signals'],
   raw_reviews_SHA256={name:sha(a.output/name) for name in ('capacity-independent-review.json','rewind-independent-review.json','graph-independent-review.json')},
   native_gate_SHA256=sha(a.native_job/'native-gate.json'),full_model_qualified=False,main_candidate_accepted=False,
   remaining_gates=['full HTTP/F1 correctness','full HTTP overhead and allocator memory','main comparison'])
  (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  state('full-storage-rewind-graph-collected-pending-HTTP-F1-and-main',positive_performance_signals=result['positive_performance_signals'])
 except Exception as error:
  state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
