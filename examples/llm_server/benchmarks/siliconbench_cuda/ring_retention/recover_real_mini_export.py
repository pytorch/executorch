"""Recover only a terminal mini export after a compiled kernel diagnostic gate."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['native-preparation','compiled-probe','baseline-audit','gguf','output','lock']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--model',choices=['MG','Gemma'],required=True);p.add_argument('--cache-root',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);pkg=Path(__file__).resolve().parent
 def state(s,**kw):
  t=a.output/'progress.pending.json';t.write_text(json.dumps(dict(status=s,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_models_paused=True,mini_accepted=False,**kw),indent=2)+'\n');t.replace(a.output/'progress.json')
 try:
  state('waiting-compiled-ring-gate');deadline=time.monotonic()+3600
  while True:
   q=json.loads(a.compiled_probe.read_text())
   if q['status']=='kernel-prerequisite-collected':break
   assert q['status']!='failed-kernel-prerequisite',q.get('error')
   assert time.monotonic()<deadline;time.sleep(20)
  assert q['compiled_journal_gate'] and len(q['results'])==6
  assert q['flat_history_bindings']
  assert q['kernel_source_SHA256']==sha(pkg/'two_source_ring.py')
  oldgate=json.loads((a.native_preparation/'native-gate.json').read_text());oldsource=Path(oldgate['export_source'])
  state('preparing-isolated-export-source-with-kernel-fix')
  source=a.output/'export-source/executorch'
  shutil.copytree(oldsource,source,symlinks=True,ignore=shutil.ignore_patterns('.git','build','cmake-out','__pycache__','*.pte','*.ptd','*.gguf','*.log'))
  target=source/'backends/cuda/triton/kernels/two_source_ring.py';before=sha(target);shutil.copy2(pkg/'two_source_ring.py',target)
  auditfile=source/'ring-retention-source-audit.json';audit=json.loads(auditfile.read_text());prior=sha(auditfile)
  name='backends/cuda/triton/kernels/two_source_ring.py';maps=[audit[k] for k in ['candidate_SHA256','additional_candidate_SHA256'] if name in audit.get(k,{})]
  assert maps,'Ring kernel is missing from original source audit'
  for hashes in maps:
   assert hashes[name]==before,'Original audited ring source differs from copied source'
   hashes[name]=sha(target)
  lower_name='backends/cuda/passes/lower_offgraph_kv.py';lower=source/lower_name;lower_before=sha(lower)
  lower_maps=[audit[k] for k in ['candidate_SHA256','additional_candidate_SHA256'] if lower_name in audit.get(k,{})]
  assert lower_maps and all(hashes[lower_name]==lower_before for hashes in lower_maps)
  text=lower.read_text();old_call='ring_commit_journal(kh,vh,k,v,info,window,stage,ring_index)';new_call='ring_commit_journal(ks,vs,k,v,info,window,stage,ring_index)'
  if old_call in text:
   assert text.count(old_call)==1;lower.write_text(text.replace(old_call,new_call))
  else:assert text.count(new_call)==1
  for hashes in lower_maps:hashes[lower_name]=sha(lower)
  audit['candidate']=str(source)
  audit['compiled_scale_recovery']=dict(before_SHA256=before,after_SHA256=sha(target),prior_audit_SHA256=prior,compiled_gate_SHA256=sha(a.compiled_probe),original_source_unchanged=True,native_runtime_unchanged=True)
  audit['raw_history_binding_recovery']=dict(before_lowering_SHA256=lower_before,after_lowering_SHA256=sha(lower),history_views_read_only=True,commit_mutates_original_flat_buffers=True,real_native_qualification_pending=True)
  auditfile.write_text(json.dumps(audit,indent=2)+'\n')
  prep=a.output/'native-preparation';prep.mkdir();gate=dict(oldgate,export_source=str(source),source_only_kernel_recovery=True,reused_native_gate=str(a.native_preparation/'native-gate.json'),reused_native_gate_SHA256=sha(a.native_preparation/'native-gate.json'),compiled_kernel_gate_SHA256=sha(a.compiled_probe),export_source_audit_SHA256=sha(auditfile))
  (prep/'native-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
  argv=[sys.executable,'-B','-u',str(pkg/'export_real_mini_pair.py'),'--native-preparation',str(prep),'--gguf',str(a.gguf),'--output',str(a.output/'exports'),'--lock',str(a.lock),'--model',a.model,'--reuse-baseline-audit',str(a.baseline_audit)]
  if a.cache_root:argv+=['--cache-root',str(a.cache_root)]
  state('exporting-ring-only-with-immutable-baseline-reuse')
  with (a.output/'export.log').open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=10800,env=dict(os.environ,CUDA_LAUNCH_BLOCKING='1'))
  state('measuring-four-arm-real-minis')
  argv=[sys.executable,'-B','-u',str(pkg/'run_real_mini_four_arms.py'),'--native-preparation',str(prep),'--export-job',str(a.output/'exports'),'--output',str(a.output/'four-arms'),'--lock',str(a.lock)]
  env=dict(os.environ);env.pop('CUDA_LAUNCH_BLOCKING',None)
  with (a.output/'four-arm.log').open('x') as f:subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=10800)
  state('real-mini-data-complete-review-required')
 except Exception as e:(a.output/'exception.txt').write_text(traceback.format_exc());state('failed-preserve-evidence',error=str(e)[-2000:]);raise
if __name__=='__main__':main()
