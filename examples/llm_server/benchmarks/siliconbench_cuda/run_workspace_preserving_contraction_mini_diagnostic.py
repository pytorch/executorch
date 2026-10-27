"""Test shared workspace preservation during independent KV contraction.

Compare with the preceding qualified private-KV SDK on identical mini PTE/PTD.
Private KV shrink, owner capacities, sliding/copies and graph invalidation are
unchanged. Four-round ABBA, capacity, rewind and actual post-shrink graph replay
remain gates. This diagnostic does not deploy or qualify a full model.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native-job','baseline-native-job','before-preparation','before-rewind-probe','lock','nsys','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,
   controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
  f.replace(a.output/'progress.json')
 def run(label,script,args):
  argv=[sys.executable,'-B','-u',str(package/script),*map(str,args)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:
   subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=14400)
 try:
  until=time.monotonic()+4*3600
  alias=a.native_job/'same-artifact-mini-preparation'
  while True:
   d=json.loads((a.native_job/'progress.json').read_text())
   if d['status']=='native-GPU-tests-passed' and (alias/'progress.json').is_file():break
   assert d['status']!='failed',d
   handle=Path('/proc')/str(d['controller_pid'])/'cmdline';actual=handle.read_bytes() if handle.is_file() else b''
   assert str(a.native_job).encode() in actual and b'prepare_independent_session_coalesced_growth_trim.py' in actual
   assert time.monotonic()<until,'Preserve verified-live build after timeout'
   state('waiting-confirmed-live-workspace-preserving-contraction-native',native_PID=d['controller_pid'],native_status=d['status'])
   time.sleep(15)
  before=verify_runtime(a.baseline_native_job,package);after=verify_runtime(a.native_job,package)
  assert after['workspace_preserving_contraction_spec_SHA256']==sha(package/'independent_session_workspace_preserving_contraction_source.json')
  assert after['preserved_preceding_native_gate_SHA256']==sha(a.baseline_native_job/'native-gate.json')
  run('rewind-probe','prepare_rewind_kv_probe.py',['--preparation-job',alias,'--python',sys.executable,'--output',a.output/'rewind-probe'])
  state('collecting-workspace-preserving-contraction-same-PTE-ABBA')
  run('matched-control','run_interleaved_native_clock_control.py',[
   '--before-preparation',a.before_preparation,'--after-preparation',alias,
   '--before-rewind-probe',a.before_rewind_probe,'--after-rewind-probe',a.output/'rewind-probe',
   '--mini-profile',a.mini_profile,'--lock',a.lock,'--nsys',a.nsys,'--output',a.output/'matched-control'])
  common=['--preparation-job',alias,'--mini-profile',a.mini_profile,'--lock',a.lock]
  state('collecting-workspace-preserving-contraction-capacity-and-rewind')
  run('capacity-control','run_independent_session_mini_capacity_control.py',common+['--output',a.output/'capacity-control'])
  run('capacity-review','review_independent_session_same_PTE_capacity.py',['--job',a.output/'capacity-control','--output',a.output/'capacity-review.json'])
  run('rewind','run_independent_session_same_artifact_control.py',common+[
   '--rewind-probe-job',a.output/'rewind-probe','--output',a.output/'rewind'])
  run('rewind-review','review_independent_session_actual_rewind.py',[
   '--job',a.output/'rewind','--profile',a.mini_profile,'--output',a.output/'rewind-review.json'])
  artifact=alias/'mini-independent-artifact';audit=json.loads((artifact/'independent-mini-artifact-audit.json').read_text())
  state('collecting-workspace-preserving-contraction-post-reclaim-graphs')
  run('graphs','run_independent_session_mini_probe.py',[
   '--probe',alias/'model-build/muse_glimmer_runtime_kv_probe','--native-gate',alias/'native-gate.json',
   '--artifact',artifact,'--mini-profile',a.mini_profile,'--arm','independent','--automatic-reclaim',
   '--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
   '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
   '--job-dir',a.output/'graphs','--lock',a.lock,'--nsys',a.nsys])
  trace=a.output/'graphs/round-1/cuda-graphs.nsys-rep';sqlite=trace.with_suffix('.sqlite');expected=sha(trace)
  with (a.output/'graph-export.log').open('x') as log:
   subprocess.run([str(a.nsys),'export','--type=sqlite','--output='+str(sqlite),str(trace)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  assert sha(trace)==expected
  run('graph-review','review_independent_session_mini_graph_trace.py',[
   '--runtime-job',a.output/'graphs','--sqlite',sqlite,'--output',a.output/'graph-review.json'])
  capacity=json.loads((a.output/'capacity-review.json').read_text());graph=json.loads((a.output/'graph-review.json').read_text())
  assert capacity['bitwise_allocation_equivalence_verified']
  assert graph['all_seven_static_methods_after_shrink_verified'] and graph['automatic_reclaim_verified_without_compact_calls']
  assert verify_runtime(a.baseline_native_job,package)==before and verify_runtime(a.native_job,package)==after
  matched=json.loads((a.output/'matched-control/result.json').read_text())
  (a.output/'result.json').write_text(json.dumps(dict(mini_profile=a.mini_profile,
   compared_to_previous_qualified_private_KV_runtime=True,workspace_preserving_contraction_candidate=True,initial_KV_capacity=512,private_KV_growth_slack_policy_unchanged=True,
   private_KV_contraction_trim_policy_unchanged=True,positive_performance_signals=matched['positive_performance_signals'],
   observed_storage_and_graph_correctness_pass=True,HTTP_not_collected=True,
   full_model_export_for_qualification_allowed=False,full_model_qualified=False,main_candidate_accepted=False,
   raw_reviews_SHA256={str(x):sha(x) for x in [a.output/'matched-control/result.json',a.output/'capacity-review.json',a.output/'rewind-review.json',a.output/'graph-review.json']}),indent=2)+'\n')
  state('workspace-preserving-contraction-mini-diagnostic-collected',positive_performance_signals=matched['positive_performance_signals'])
 except Exception as error:
  state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
