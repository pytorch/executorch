"""Collect twenty Gemma main cases only after complete raw-bound qualification.

Keep all original gates. Follow the actual full quality controller; correctness,
matching native controls and memory are reverified before main collection.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('quality-follower-job','full-controls-job','native-job','mini-qualification','memory-worker-proof','Gemma-source-preparation-job','http-python-dependencies-job','tokenizer','server-python','public-inputs','lock','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--port',type=int,required=True)
 p.add_argument('--matched-confirmation-job',type=Path)
 p.add_argument('--performance-acceptance',type=Path)
 a=p.parse_args();assert 1024<=a.port<=65535 and not a.output.exists();a.output.mkdir(parents=True)
 assert not a.performance_acceptance or a.matched_confirmation_job
 package=Path(__file__).resolve().parent
 def state(status,**kw):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),full_model_qualified=False,main_candidate_accepted=False,**kw),indent=2)+'\n');f.replace(a.output/'progress.json')
 def run(label,script,argv,timeout):
  command=[str(a.server_python),'-B','-u',str(package/script),*map(str,argv)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=command),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:subprocess.run(command,check=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 try:
  deadline=time.monotonic()+36*3600
  while True:
   d=json.loads((a.quality_follower_job/'progress.json').read_text())
   if d['status']=='fresh-full-Gemma-native-ABBA-and-quality-collected':break
   assert d['status']!='failed',d
   proc=Path('/proc')/str(d['controller_pid'])/'cmdline';actual=proc.read_bytes().split(bytes([0])) if proc.is_file() else []
   assert any(Path(x.decode()).name=='run_full_Gemma_quality_after_allocator_controls.py' for x in actual if x)
   assert b'--output' in actual and actual[actual.index(b'--output')+1]==os.fsencode(a.quality_follower_job)
   assert time.monotonic()<deadline,'Preserve verified-live quality controller after observation timeout'
   state('waiting-confirmed-live-fresh-full-Gemma-quality',parent_PID=d['controller_pid'],parent_status=d['status']);time.sleep(20)
  parent=a.quality_follower_job/'result.json';parent_hash=sha(parent);doc=json.loads(parent.read_text())
  assert doc['raw_full_quality_pass'] and (a.matched_confirmation_job or not doc['positive_performance_signals'])
  if a.matched_confirmation_job:
   while True:
    d=json.loads((a.matched_confirmation_job/'progress.json').read_text())
    if d['status']=='interleaved-native-control-collected':break
    assert d['status']!='failed',d
    proc=Path('/proc')/str(d['controller_pid'])/'cmdline';actual=proc.read_bytes().split(bytes([0])) if proc.is_file() else []
    assert any(Path(x.decode()).name=='run_interleaved_native_clock_control.py' for x in actual if x)
    assert b'--output' in actual and actual[actual.index(b'--output')+1]==os.fsencode(a.matched_confirmation_job)
    assert time.monotonic()<deadline
    state('waiting-live-full-native-ABBA-confirmation',confirmation_PID=d['controller_pid'],confirmation_status=d['status']);time.sleep(20)
  assert not doc['full_model_qualified'] and not doc['main_candidate_accepted']
  common=[]
  for n in ('native-job','mini-qualification','memory-worker-proof','Gemma-source-preparation-job','http-python-dependencies-job','tokenizer','public-inputs'):common+=['--'+n,getattr(a,n.replace('-','_'))]
  confirmation_args=['--matched-confirmation-job',a.matched_confirmation_job] if a.matched_confirmation_job else []
  if a.performance_acceptance:confirmation_args+=['--performance-acceptance',a.performance_acceptance]
  state('rechecking-complete-full-Gemma-qualification')
  run('full-qualification','write_Gemma_full_independent_KV_qualification.py',['--package',package,*common,'--full-controls-job',a.full_controls_job,'--quality-follower-job',a.quality_follower_job,'--output',a.output/'full-qualification',*confirmation_args],4*3600)
  receipt=a.output/'full-qualification/full-qualification.json';qualified=json.loads(receipt.read_text())
  assert qualified['full_model_qualified'] and not qualified['main_candidate_accepted'] and not qualified['unresolved_correctness_or_performance_signals']
  assert sha(parent)==parent_hash
  state('collecting-qualified-full-Gemma-main',full_qualification_receipt_SHA256=sha(receipt))
  run('main-performance','run_gemma_independent_full_main.py',['--phase','performance','--full-model-profile','gemma4-48layers','--package',package,*common,'--fixed-job',a.full_controls_job/'full-fixed','--HTTP-job',a.full_controls_job/'full-HTTP','--server-python',a.server_python,'--lock',a.lock,'--port',a.port,'--full-qualification-receipt',receipt,'--output',a.output/'main-performance'],36*3600)
  collected=json.loads((a.output/'main-performance/result.json').read_text());assert len(collected['completed'])==20
  assert collected['raw_main_acceptance_pending'] and not collected['main_candidate_accepted']
  assert sha(parent)==parent_hash
  (a.output/'result.json').write_text(json.dumps(dict(full_profile='gemma4-48layers',main_cases_collected=20,qualification_SHA256=sha(receipt),quality_follower_result_SHA256=parent_hash,main_result_SHA256=sha(a.output/'main-performance/result.json'),full_model_qualified=True,main_candidate_accepted=False,raw_main_review_and_same_GPU_accepted_baseline_comparison_pending=True),indent=2)+'\n')
  state('qualified-Gemma-main-collected-pending-independent-review')
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
