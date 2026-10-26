"""Build the exact existing private allocator chain on another mini baseline.

Use the existing immutable mini PTE/PTD and matched probe on this GPU. Four
native transitions require actual CUDA tests. This checks host/baseline effects;
it never deploys a full model or accepts main results.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path
from independent_session_full_native_binding import sha,verify_runtime


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('baseline-native-job','before-preparation','before-rewind-probe','lock','nsys','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**kw):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),full_model_qualified=False,main_candidate_accepted=False,**kw),indent=2)+'\n');f.replace(a.output/'progress.json')
 def run(label,script,argv):
  cmd=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=cmd),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:subprocess.run(cmd,check=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=4*3600)
 try:
  baseline=verify_runtime(a.baseline_native_job,package);before_hash=sha(a.baseline_native_job/'native-gate.json')
  assert (a.before_preparation/'native-gate.json').is_file()
  jobs=[];current=a.baseline_native_job
  for policy in ('workspace-preserving','isolated-KV','isolated-KV-growth-slack','paired-KV-storage'):
   output=a.output/(policy+'-native');state('building-exact-source-native-transition',policy=policy)
   run(policy,'prepare_independent_session_coalesced_growth_trim.py',[
    '--package',package,'--baseline-native-job',current,'--allocator-policy',policy,
    '--source-variant','Gemma-empty-owner-reuse' if a.mini_profile=='gemma4-first6' else 'MG-empty-owner-reuse',
    '--mini-profile',a.mini_profile,'--mini-preparation-job',a.before_preparation,'--jobs','4',
    '--python',sys.executable,'--lock',a.lock,'--output',output])
   verify_runtime(output,package);jobs.append(output);current=output
  state('collecting-original-baseline-paired-mini-controls')
  run('mini','run_paired_KV_storage_mini_diagnostic.py',[
   '--native-job',jobs[-1],'--baseline-native-job',jobs[-2],
   '--before-preparation',a.before_preparation,'--before-rewind-probe',a.before_rewind_probe,
   '--mini-profile',a.mini_profile,'--lock',a.lock,'--nsys',a.nsys,'--output',a.output/'mini-diagnostic'])
  assert sha(a.baseline_native_job/'native-gate.json')==before_hash
  assert verify_runtime(a.baseline_native_job,package)==baseline
  result=json.loads((a.output/'mini-diagnostic/result.json').read_text())
  (a.output/'result.json').write_text(json.dumps(dict(initial_baseline_native_job=str(a.baseline_native_job),initial_native_gate_SHA256=before_hash,actual_intermediate_native_gates={str(x):sha(x/'native-gate.json') for x in jobs},positive_performance_signals=result['positive_performance_signals'],full_model_qualified=False,main_candidate_accepted=False,scope='Existing-host mini baseline replication; compare only this actual baseline, not another host or a prior main runtime.'),indent=2)+'\n')
  state('paired-chain-and-mini-controls-collected',positive_performance_signals=result['positive_performance_signals'])
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
