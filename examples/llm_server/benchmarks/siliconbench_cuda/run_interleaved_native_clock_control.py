"""Resolve a native timing signal with matched probes, warmup and ABBA order.

Read clocks/power every second without changing GPU settings. Keep all earlier
runs and failed gates. Four rounds, discard first, preserve actual model inputs,
tensors and child provenance. This diagnostic never promotes a full model.
"""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from qualified_rewind_probe import qualified_rewind_probe


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('before-preparation','after-preparation','before-rewind-probe','after-rewind-probe','lock','nsys','output'):
  p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mini-profile',choices=('muse-glimmer-first4','gemma4-first6'),default='muse-glimmer-first4')
 p.add_argument('--diagnostic-initial-KV-capacity',type=int,choices=(512,16384),default=512,
  help='Same capacity in both native arms;16384 is allocation-placement diagnostic only, never a replacement main preset')
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_candidate_accepted=False,
   full_model_qualified=False,**extra),indent=2)+'\n');f.replace(a.output/'progress.json')
 stop=threading.Event();current={'phase':'preflight'};sample_errors=[]
 def observe():
  with (a.output/'GPU-clocks-power.jsonl').open('x') as f:
   while not stop.is_set():
    try:
     text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
     f.write(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic(),phase=dict(current),actual_GPU_query=text))+'\n');f.flush()
    except Exception as e:sample_errors.append(str(e))
    stop.wait(1)
 thread=None
 try:
  preparations={'before':a.before_preparation,'after':a.after_preparation}
  probes={'before':a.before_rewind_probe,'after':a.after_rewind_probe};gates={};proofs={}
  for arm,prep in preparations.items():
   assert json.loads((prep/'progress.json').read_text())['status']=='mini-artifacts-ready'
   gate=prep/'native-gate.json';gates[arm]=sha(gate)
   _,proofs[arm]=qualified_rewind_probe(probes[arm],gate,prep/'model-build/muse_glimmer_runtime_kv_probe',state,0)
  assert proofs['before']['generated_CPP_SHA256']==proofs['after']['generated_CPP_SHA256']
  for layout in ('global','independent'):
   audits=[json.loads((prep/f'mini-{layout}-artifact/independent-mini-artifact-audit.json').read_text()) for prep in preparations.values()]
   from independent_session_mini_profiles import validate_mini_audit
   for audit in audits:validate_mini_audit(audit,a.mini_profile,layout=='independent')
   assert audits[0]['artifacts']==audits[1]['artifacts']
  thread=threading.Thread(target=observe,daemon=True);thread.start();parts=[]
  warm_code="""import time,torch
assert torch.cuda.get_device_capability()==(12,0)
x=torch.randn((8192,8192),device='cuda',dtype=torch.bfloat16)
y=torch.randn_like(x);z=torch.empty_like(x)
end=time.monotonic()+20
while time.monotonic()<end:
 for _ in range(16):torch.mm(x,y,out=z)
 torch.cuda.synchronize()
"""
  for number in range(1,5):
   for arm in (('before','after') if number%2 else ('after','before')):
    current.update(phase='warmup',round=number,native=arm);state('warming-GPU-for-matched-native-control',round=number,native=arm)
    with a.lock.open('a') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX)
     assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
     with (a.output/f'warmup-round-{number}-{arm}.log').open('x') as log:
      subprocess.run([sys.executable,'-B','-c',warm_code],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120,
       env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2'})
     until=time.monotonic()+120
     while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
      assert time.monotonic()<until;time.sleep(1)
    child=a.output/f'child-round-{number}-{arm}'
    command=[sys.executable,'-B','-u',str(package/'run_independent_session_mini_model_latency.py'),
     '--preparation-job',str(preparations[arm]),'--rewind-probe-job',str(probes[arm]),
     '--native-sampler','off','--mini-profile',a.mini_profile,
     '--diagnostic-initial-KV-capacity',str(a.diagnostic_initial_KV_capacity),
     '--diagnostic-round-number',str(number),'--nsys',str(a.nsys),'--lock',str(a.lock),'--output',str(child)]
    current.update(phase='fixed-token-collection',round=number,native=arm);state('collecting-interleaved-native-control',round=number,native=arm)
    with (a.output/f'child-round-{number}-{arm}.log').open('x') as log:
     subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=3600)
    doc=json.loads((child/'result.json').read_text());assert doc['diagnostic_single_round']==number and len(doc['rounds'])==2
    assert not doc['model_performance_qualified'] and not doc['candidate_promoted']
    assert doc['native_gate_sha256']==gates[arm]
    assert {r['round'] for r in doc['rounds']}=={number}
    assert {r['arm'] for r in doc['rounds']}=={'global','independent'}
    parts.append(dict(native=arm,round=number,result=str(child/'result.json'),SHA256=sha(child/'result.json'),command=command))
  combined={}
  for arm in ('before','after'):
   dest=a.output/arm;dest.mkdir();selected=[r for r in parts if r['native']==arm]
   assert [r['round'] for r in selected]==[1,2,3,4]
   docs=[json.loads(Path(r['result']).read_text()) for r in selected]
   result={k:v for k,v in docs[0].items() if k not in ('rounds','aggregates','diagnostic_single_round','actual_profiled_rounds')}
   result.update(rounds=[r for d in docs for r in d['rounds']],aggregates=[],
    diagnostic_single_round=None,four_rounds_first_discarded=True,actual_profiled_rounds=[1,1],
    interleaved_native_round_order=True,all_actual_child_results_verified=selected)
   for r in selected:
    child=Path(r['result']).parent;(dest/f"round-{r['round']}").symlink_to(child/f"round-{r['round']}")
   (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n');combined[arm]=dest
  for arm in ('before','after'):
   assert sha(preparations[arm]/'native-gate.json')==gates[arm]
  state('independently-reviewing-matched-native-control')
  review=a.output/'same-PTE-native-independent-review.json'
  with (a.output/'independent-review.log').open('x') as log:
   subprocess.run([sys.executable,'-B',str(package/'review_independent_session_same_artifact_native_change.py'),
    '--before-job',str(combined['before']),'--after-job',str(combined['after']),'--output',str(review)],
    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  r=json.loads(review.read_text());signals=[{'arm':c['arm'],'case':c['case'],'metric':m,**c[m]} for c in r['cells'] for m in ('cold_first_four_seconds','mean_stable_native_ms') if c[m]['positive_signal']]
  stop.set();thread.join(timeout=20)
  assert not sample_errors,sample_errors
  result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   mini_profile=a.mini_profile,initial_KV_capacity=a.diagnostic_initial_KV_capacity,
   four_rounds_discard_first=True,ABBA_native_order=True,matched_probe_CPP=True,
   unchanged_PTE_PTD_and_tokens=True,native_tests_and_proofs_reverified=True,
   actual_child_results=parts,raw_numerical_review_SHA256=sha(review),
   clock_power_samples_SHA256=sha(a.output/'GPU-clocks-power.jsonl'),positive_performance_signals=signals,
   previous_failed_gate_not_replaced=True,GPU_clock_settings_changed=False,
   full_model_qualified=False,main_candidate_accepted=False,
   scope='Fixed-token diagnostic with per-process warmup and readonly clock/power samples. Cold changes, natural task throughput/F1, main acceptance and a causal explanation remain separate.')
  (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('interleaved-native-control-collected',positive_performance_signals=signals)
 except Exception as e:
  state('failed',error=str(e),traceback=traceback.format_exc());raise
 finally:
  stop.set()
  if thread:thread.join(timeout=20)

if __name__=='__main__':main()
