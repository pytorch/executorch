"""Review both native baselines' global/independent raw timing and capture data.

Wait for the exact detached profile controller. No GPU recollection, runtime
modifications, qualification override or main acceptance. Preserve all signals.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from independent_session_full_native_binding import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--profile-job',type=Path,required=True)
 p.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**kw):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),main_candidate_accepted=False,full_model_qualified=False,**kw),indent=2)+'\n');f.replace(a.output/'progress.json')
 try:
  deadline=time.monotonic()+5*3600
  while True:
   progress=json.loads((a.profile_job/'progress.json').read_text())
   if progress['status']=='signal-reproducibility-and-trace-review-collected':break
   assert progress['status']!='failed',progress
   proc=Path('/proc')/str(progress['controller_pid'])/'cmdline'
   args=proc.read_bytes().split(bytes([0])) if proc.is_file() else []
   assert any(Path(x.decode()).name=='run_allocator_profile_followup.py' for x in args if x)
   assert b'--output' in args and args[args.index(b'--output')+1]==os.fsencode(a.profile_job)
   assert time.monotonic()<deadline,'Preserve verified-live profile controller on observation timeout'
   state('waiting-confirmed-live-signal-profile',parent_PID=progress['controller_pid']);time.sleep(20)
  state('recomputing-global-independent-raw-layout-timing')
  matched=a.profile_job/'matched-control';profile=a.profile_job/'trace-review.json'
  trace=json.loads(profile.read_text());assert trace['matched_result_SHA256']==sha(matched/'result.json')
  assert trace['automatic_performance_waiver'] is False
  reviews={};signals=[]
  for side in ('before','after'):
   dest=a.output/(side+'-layout-fixed-latency.json')
   script='review_independent_session_gemma_fixed_latency.py' if a.mini_profile=='gemma4-first6' else 'review_independent_session_mg_fixed_latency.py'
   argv=[sys.executable,'-B',str(package/script),'--job',str(matched/side),'--output',str(dest)]
   with (a.output/(side+'-layout-review.log')).open('x') as log:subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
   review=json.loads(dest.read_text());reviews[side]=dict(path=str(dest),SHA256=sha(dest))
   for c in review['aggregates']:
    for metric in ('cold_first_four_seconds','mean_stable_native_ms'):
     v=c[metric];delta=v['arms']['independent']['mean']-v['arms']['global']['mean']
     if delta>v['combined_3SE']:signals.append(dict(native=side,comparison='global-to-independent',case=c['case'],metric=metric,delta=delta,delta_percent=v['delta_percent'],combined_3SE=v['combined_3SE']))
  layout_trace=[]
  for side,rows in trace['arms'].items():
   global_rows=[r for r in rows if r['layout']=='global'];independent_rows=[r for r in rows if r['layout']=='independent']
   assert [r['case'] for r in global_rows]==[r['case'] for r in independent_rows]
   for b,c in zip(global_rows,independent_rows):layout_trace.append(dict(native=side,case=b['case'],stable_delta={k:c['stable_means'][k]-b['stable_means'][k] for k in b['stable_means']},cold_delta={k:c['cold_sums'][k]-b['cold_sums'][k] for k in b['cold_sums']}))
  result=dict(mini_profile=a.mini_profile,actual_four_round_layout_reviews=reviews,global_independent_positive_signals=signals,same_PTE_native_positive_signals=trace['original_positive_signals'],discarded_round_layout_trace_deltas=layout_trace,trace_SHA256=sha(profile),profile_job=str(a.profile_job),automatic_performance_waiver=False,main_candidate_accepted=False,full_model_qualified=False,sums_may_overlap_not_additive_critical_path=True)
  (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('global-independent-layout-raw-timing-and-capture-reviewed',positive_performance_signals=signals)
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
