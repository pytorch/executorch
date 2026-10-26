"""Detached four-round verification followed by exact saved-call trace review.

No runtime modifications, exports or automatic gate waiver. Preserve earlier
failures. A new attempt checks signal reproducibility and capture attribution.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('before-preparation','after-preparation','before-rewind-probe','after-rewind-probe','lock','nsys','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**kw):
  dest=a.output/'progress.pending.json';dest.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),main_candidate_accepted=False,full_model_qualified=False,**kw),indent=2)+'\n');dest.replace(a.output/'progress.json')
 def run(script,argv,label):
  cmd=[sys.executable,'-B','-u',str(package/script),*map(str,argv)]
  (a.output/(label+'-command.json')).write_text(json.dumps(dict(argv=cmd),indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
 try:
  state('collecting-four-round-signal-reproducibility-and-profile')
  args=[]
  for n in ('before_preparation','after_preparation','before_rewind_probe','after_rewind_probe','lock','nsys','mini_profile'):args+=['--'+n.replace('_','-'),getattr(a,n)]
  run('run_interleaved_native_clock_control.py',args+['--output',a.output/'matched-control'],'matched-control')
  state('reviewing-exact-cold-and-stable-raw-profile')
  run('review_allocator_matched_trace.py',['--matched-job',a.output/'matched-control','--output',a.output/'trace-review.json'],'trace-review')
  result=json.loads((a.output/'matched-control/result.json').read_text())
  state('signal-reproducibility-and-trace-review-collected',positive_performance_signals=result['positive_performance_signals'],automatic_performance_waiver=False)
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
