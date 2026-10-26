"""Retain full consecutive controls and collect a separate matched ABBA trace.

Use the already mini-gated full PTE/PTD and both existing native probes.
Wait for native/HTTP memory workers to finish before starting GPU timing.
This diagnostic cannot promote a main configuration or erase a failed gate.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native-lifecycle-job','HTTP-job','native-job','lock','nsys','output'):
  p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,
   controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
  f.replace(a.output/'progress.json')
 def wait(job,terminal,controller):
  until=time.monotonic()+12*3600
  while True:
   progress=json.loads((job/'progress.json').read_text())
   if progress['status']==terminal:return progress
   assert progress['status']!='failed',progress
   handle=Path('/proc')/str(progress['controller_pid'])/'cmdline'
   actual=handle.read_bytes() if handle.is_file() else b''
   if not actual and json.loads((job/'progress.json').read_text())['status']==terminal:return
   assert str(job).encode() in actual and controller.encode() in actual
   assert time.monotonic()<until,'Preserve verified-live parent after timeout'
   state('waiting-confirmed-live-full-control',parent=str(job),parent_PID=progress['controller_pid'],parent_status=progress['status'])
   time.sleep(20)
 try:
  wait(a.native_lifecycle_job,'full-MG-single-artifact-native-controls-collected','run_single_full_MG_lifecycle_after_export.py')
  wait(a.HTTP_job,'full-HTTP-memory-collected-pending-F1-and-main','run_full_MG_HTTP_after_native.py')
  gate=verify_runtime(a.native_job,package)
  native_path=a.native_lifecycle_job/'result.json';native=json.loads(native_path.read_text())
  HTTP_path=a.HTTP_job/'result.json';HTTP=json.loads(HTTP_path.read_text())
  assert native['full_profile']==HTTP['full_profile']=='muse-glimmer-52layers'
  assert native['full_independent_storage_and_observed_numerics_pass'] and HTTP['full_HTTP_memory_pass']
  assert native['actual_runtime_native_gate_SHA256']==HTTP['native_runtime_gate_SHA256']==sha(a.native_job/'native-gate.json')
  assert HTTP['native_lifecycle_result_SHA256']==sha(native_path)
  preserved={str(native_path):sha(native_path),str(HTTP_path):sha(HTTP_path)}
  for name,expected in native['raw_reviews_SHA256'].items():
   path=a.native_lifecycle_job/name;assert sha(path)==expected;preserved[str(path)]=expected
  assert sha(a.HTTP_job/'HTTP-independent-review.json')==HTTP['HTTP_review_SHA256']
  before=a.native_lifecycle_job/'before-full-artifact';after=a.native_lifecycle_job/'after-full-artifact'
  argv=[sys.executable,'-B','-u',str(package/'run_interleaved_native_clock_control.py'),
   '--before-preparation',str(before),'--after-preparation',str(after),
   '--before-rewind-probe',str(a.native_lifecycle_job/'before-rewind-probe'),
   '--after-rewind-probe',str(a.native_lifecycle_job/'after-rewind-probe'),
   '--mini-profile','muse-glimmer-first4','--full-model-profile','muse-glimmer-52layers','--arm','independent',
   '--lock',str(a.lock),'--nsys',str(a.nsys),'--output',str(a.output/'matched-control')]
  (a.output/'matched-command.json').write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
  state('collecting-matched-full-MG-native-ABBA',preserved_consecutive_signals=native['positive_performance_signals'])
  with (a.output/'matched-control.log').open('x') as log:
   subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=21600)
  matched=json.loads((a.output/'matched-control/result.json').read_text())
  assert matched['full_model_profile']=='muse-glimmer-52layers' and matched['collected_arms']==['independent']
  assert matched['initial_KV_capacity']==512 and matched['four_rounds_discard_first'] and matched['ABBA_native_order']
  review_path=a.output/'matched-control/same-PTE-native-independent-review.json'
  review=json.loads(review_path.read_text())
  assert review['full_saved_FP32_tensors_equal']==9 and review['selected_argmax_values_equal']==3376
  assert review['native_change_preserves_observed_model_numerics']
  assert verify_runtime(a.native_job,package)==gate
  for name,expected in preserved.items():assert sha(name)==expected
  result=dict(full_profile='muse-glimmer-52layers',matched_control_result_SHA256=sha(a.output/'matched-control/result.json'),
   matched_raw_review_SHA256=sha(review_path),positive_performance_signals=matched['positive_performance_signals'],
   original_consecutive_signals=native['positive_performance_signals'],original_evidence_SHA256=preserved,
   matched_fixed_token_performance_pass=not matched['positive_performance_signals'],
   original_gates_unchanged=True,independent_local_raw_review_pending=True,
   full_task_F1_pending=True,full_model_qualified=False,main_candidate_accepted=False)
  (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  state('full-MG-matched-native-diagnostic-collected',positive_performance_signals=matched['positive_performance_signals'])
 except Exception as error:
  state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
