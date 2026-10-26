"""Retain a failed cold mini gate and verify a second complete ABBA study.

Recompute both real four-round studies, each discarding round1, with unchanged
3SE formula. Reuse previously verified HTTP/storage without recollection. No
full deployment or main acceptance is performed by this controller.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('original-HTTP-job','original-mini-diagnostic-job','confirmation-profile-job','native-job','Gemma-source-preparation-job','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**kw):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),main_candidate_accepted=False,full_model_qualified=False,**kw),indent=2)+'\n');f.replace(a.output/'progress.json')
 try:
  state('checking-actual-retained-mini-and-layout-confirmation')
  original=a.original_HTTP_job/'mini-reviews/mini-qualification.json';original_hash=sha(original);q=json.loads(original.read_text())
  assert q['mini_profile']=='gemma4-first6' and q['mini_storage_correctness_pass'] and q['mini_HTTP_memory_pass']
  assert not q['mini_cold_and_steady_performance_pass'] and q['positive_performance_signals']
  assert all(x['comparison']=='global-to-independent' and x['metric']=='cold_first_four_seconds' for x in q['positive_performance_signals'])
  assert not q['full_model_qualified'] and not q['main_candidate_accepted'] and not q['full_model_export_for_qualification_allowed']
  assert q['runtime_native_gate_SHA256']==sha(a.native_job/'native-gate.json')
  for name,expected in q['evidence_files_SHA256'].items():assert sha(name)==expected,name
  verify_runtime(a.native_job,package)
  confirmation=a.confirmation_profile_job
  status=json.loads((confirmation/'progress.json').read_text());assert status['status']=='signal-reproducibility-and-trace-review-collected'
  matched=confirmation/'matched-control';result=json.loads((matched/'result.json').read_text())
  assert not result['positive_performance_signals'],result['positive_performance_signals']
  assert json.loads((a.original_mini_diagnostic_job/'progress.json').read_text())['status']=='paired-KV-storage-mini-diagnostic-collected'
  assert (a.original_HTTP_job/'mini-controls').is_dir()
  (a.output/'mini-controls').symlink_to((a.original_HTTP_job/'mini-controls').resolve())
  state('recomputing-two-actual-mini-ABBA-triplicates-and-all-raw-gates')
  argv=[sys.executable,'-B',str(package/'qualify_independent_session_mini_for_full.py'),
   '--mini-job',str(a.original_HTTP_job/'mini-controls'),'--native-job',str(a.native_job),
   '--previous-latency-job',str(a.original_mini_diagnostic_job/'matched-control/before'),
   '--profile','gemma4-first6','--Gemma-source-preparation-job',str(a.Gemma_source_preparation_job),
   '--matched-native-control-job',str(a.original_mini_diagnostic_job/'matched-control'),
   '--preserved-native-controls-job',str(a.original_mini_diagnostic_job),
   '--additional-matched-layout-control-job',str(matched),'--output',str(a.output/'mini-reviews')]
  (a.output/'qualification-command.json').write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
  with (a.output/'qualification.log').open('x') as log:subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
  receipt=json.loads((a.output/'mini-reviews/mini-qualification.json').read_text())
  assert sha(original)==original_hash
  passed=receipt['full_model_export_for_qualification_allowed']
  (a.output/'result.json').write_text(json.dumps(dict(original_failed_qualification_SHA256=sha(original),new_qualification_SHA256=sha(a.output/'mini-reviews/mini-qualification.json'),full_model_export_for_qualification_allowed=passed,positive_performance_signals=receipt['positive_performance_signals'],full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
  state('mini-qualified-pending-full-model' if passed else 'mini-performance-gate-not-passed',positive_performance_signals=receipt['positive_performance_signals'])
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
