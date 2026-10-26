"""Collect and independently score new MG F1 after the real matched control.

Keep original consecutive timing signals. A failed matched control prevents
this fresh full F1 collection; no file or runtime is relabelled as qualified.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('matched-native-job','HTTP-job','native-job','native-lifecycle-job','mini-qualification',
              'memory-worker-proof','http-python-dependencies-job','tokenizer','server-python',
              'public-inputs','lock','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--port',type=int,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,
   controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
  f.replace(a.output/'progress.json')
 def run(label,script,args,timeout):
  argv=[str(a.server_python),'-B','-u',str(package/script),*map(str,args)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:
   subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=timeout)
 try:
  until=time.monotonic()+12*3600
  while True:
   d=json.loads((a.matched_native_job/'progress.json').read_text())
   if d['status']=='full-MG-matched-native-diagnostic-collected':break
   assert d['status']!='failed',d
   handle=Path('/proc')/str(d['controller_pid'])/'cmdline';actual=handle.read_bytes() if handle.is_file() else b''
   if not actual and json.loads((a.matched_native_job/'progress.json').read_text())['status']=='full-MG-matched-native-diagnostic-collected':break
   assert str(a.matched_native_job).encode() in actual and b'run_full_MG_matched_native_after_controls.py' in actual
   assert time.monotonic()<until,'Preserve verified-live parent after timeout'
   state('waiting-confirmed-live-full-MG-matched-native',parent_PID=d['controller_pid'],parent_status=d['status'])
   time.sleep(20)
  gate=verify_runtime(a.native_job,package)
  path=a.matched_native_job/'result.json';matched=json.loads(path.read_text())
  assert matched['full_profile']=='muse-glimmer-52layers'
  assert matched['original_gates_unchanged'] and not matched['full_model_qualified']
  assert not matched['main_candidate_accepted']
  for name,expected in matched['original_evidence_SHA256'].items():assert sha(name)==expected,name
  native_result=a.native_lifecycle_job/'result.json'
  assert matched['original_evidence_SHA256'][str(native_result)]==sha(native_result)
  assert sha(a.matched_native_job/'matched-control/result.json')==matched['matched_control_result_SHA256']
  assert sha(a.matched_native_job/'matched-control/same-PTE-native-independent-review.json')==matched['matched_raw_review_SHA256']
  if matched['positive_performance_signals']:
   assert not matched['matched_fixed_token_performance_pass']
   (a.output/'result.json').write_text(json.dumps(dict(fresh_F1_collected=False,
    matched_result_SHA256=sha(path),positive_performance_signals=matched['positive_performance_signals'],
    full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
   state('full-performance-gate-not-passed',positive_performance_signals=matched['positive_performance_signals'])
   return
  assert matched['matched_fixed_token_performance_pass']
  state('collecting-fresh-full-MG-task-quality',matched_result_SHA256=sha(path))
  args=['--phase','quality','--full-model-profile','muse-glimmer-52layers','--package',package]
  for name in ('HTTP-job','native-job','native-lifecycle-job','mini-qualification','memory-worker-proof',
               'http-python-dependencies-job','tokenizer','server-python','public-inputs','lock'):
   args.extend(['--'+name,getattr(a,name.replace('-','_'))])
  args.extend(['--port',str(a.port),'--output',a.output/'full-quality'])
  run('full-quality','run_gemma_independent_full_main.py',args,48*3600)
  state('independently-reviewing-fresh-full-MG-task-quality')
  run('raw-quality-review','review_independent_full_task_quality.py',[
   '--quality-job',a.output/'full-quality','--public-inputs',a.public_inputs,'--native-job',a.native_job,
   '--full-model-profile','muse-glimmer-52layers','--output',a.output/'raw-quality-review.json'],3600)
  review=json.loads((a.output/'raw-quality-review.json').read_text())
  assert review['complete'] and review['reviewed_shots']==20 and review['actual_response_count']==9936
  assert review['full_task_F1_independently_reviewed'] and review['full_task_raw_quality_pass']
  assert review['native_runtime_gate_SHA256']==sha(a.native_job/'native-gate.json')
  assert sha(path)==sha(a.matched_native_job/'result.json')
  assert verify_runtime(a.native_job,package)==gate
  (a.output/'result.json').write_text(json.dumps(dict(fresh_F1_collected=True,
   full_task_F1_independently_reviewed=True,full_task_raw_quality_pass=True,
   native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),raw_quality_review_SHA256=sha(a.output/'raw-quality-review.json'),
   full_model_qualified=False,main_candidate_accepted=False,
   remaining_gates=['independent local full raw review','full qualification receipt','main performance/memory comparison']),indent=2)+'\n')
  state('fresh-full-MG-quality-collected-and-recomputed')
 except Exception as error:
  state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
