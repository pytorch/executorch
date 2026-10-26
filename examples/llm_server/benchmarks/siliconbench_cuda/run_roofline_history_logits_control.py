"""Isolate prefill history and compare teacher-forced full model logits.

Use equal fixed future tokens; do not conflate autoregressive divergence with
per-forward numerical error. This diagnostic never promotes a timing or F1.
"""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(16*1024**2):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['workflow','ET-SDK-job','et-source','lock','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--deadline-UTC',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 deadline=datetime.datetime.fromisoformat(a.deadline_UTC);assert deadline.tzinfo
 package=Path(__file__).resolve().parent
 def remaining():
  v=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert v>0;return v
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_only=True,**extra),indent=2)+'\n');f.replace(a.output/'progress.json')
 try:
  sdk=json.loads((a.ET_SDK_job/'ET-provenance.json').read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
  prefix=subprocess.check_output([sys.executable,'-B','-c','import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
  build=a.output/'build';state('building-CPU-logit-control-driver')
  with (a.output/'build.log').open('x') as log:
   for cmd in [['cmake','-S',package/'roofline_validation_project','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DET_SOURCE='+str(a.et_source),'-DCMAKE_PREFIX_PATH='+sdk['SDK']+';'+prefix],['cmake','--build',build,'--parallel','2','--target','roofline_prefill_validation']]:subprocess.run([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
  provenance=json.loads((a.workflow/'full-model-comparison/input-provenance.json').read_text());arms=provenance['artifact_arms'];assert {r['name'] for r in arms}=={'ET-baseline','ET-large'}
  assert arms[0]['hashes']['aoti_cuda_blob.ptd']==arms[1]['hashes']['aoti_cuda_blob.ptd']
  env['CUDA_VISIBLE_DEVICES']='0';records=[];binary=build/'roofline_prefill_validation'
  with a.lock.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   for round_id in [1,2,3,4]:
    for chunk in [1024,512]:
     for arm in arms:
      assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
      state('running-history-and-teacher-forced-logit-control',round=round_id,history_chunk=chunk,arm=arm['name'])
      output=a.output/(arm['name']+'-history'+str(chunk)+'-round'+str(round_id)+'.json');artifact=Path(arm['artifact'])
      with output.with_suffix('.log').open('x') as log:subprocess.run([str(binary),str(artifact/'model.pte'),str(artifact/'aoti_cuda_blob.ptd'),str(output),str(chunk),'1','1'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
      value=json.loads(output.read_text());assert value['completed'] and value['teacher_forced'] and len(value['cases'])==7
      records.append(dict(round=round_id,history_chunk=chunk,arm=arm['name'],file=output.name,SHA256=sha(output)))
  state('computing-local-CPU-numerical-differences')
  def load(row):
   f=Path(row['file']);format=row['format'];dtype={'bf16':'<u2','f16':'<f2','f32':'<f4'}[format]
   x=np.fromfile(f,dtype=dtype)
   if format=='bf16':x=(x.astype(np.uint32)<<16).view(np.float32)
   else:x=x.astype(np.float32)
   assert x.size==row['rows']*row['vocab'];return x.reshape(row['rows'],row['vocab'])
  comparison=[];files={};summaries={}
  for record in records:
   name=record['file'];x=json.loads((a.output/name).read_text());summaries[record['arm'],record['history_chunk'],record['round']]=x
   files[name]=dict(bytes=(a.output/name).stat().st_size,SHA256=sha(a.output/name))
   for c in x['cases']:
    assert len(c['logit_dumps'])==5
    for dump in c['logit_dumps']:
     f=Path(dump['file']);files[f.name]=dict(bytes=f.stat().st_size,SHA256=sha(f))
  for chunk in [1024,512]:
   for rid in [1,2,3,4]:
    for gold_case,test_case in zip(summaries['ET-baseline',chunk,rid]['cases'],summaries['ET-large',chunk,rid]['cases']):
     assert gold_case['mix']==test_case['mix'];steps=[]
     for step,(gold_dump,test_dump) in enumerate(zip(gold_case['logit_dumps'],test_case['logit_dumps'])):
      assert gold_dump['rows']==test_dump['rows'] and gold_dump['vocab']==test_dump['vocab']
      gold=load(gold_dump);test=load(test_dump);assert np.isfinite(gold).all() and np.isfinite(test).all()
      diff=gold.astype(np.float64)-test;den=np.sum(gold.astype(np.float64)**2);rms=float(np.sqrt(np.sum(diff**2)/max(den,1e-30)))
      top_gold=np.argmax(gold,axis=1);top_test=np.argmax(test,axis=1);mismatch=np.flatnonzero(top_gold!=top_test);margins=[]
      for row in mismatch:
       two=np.partition(gold[row],-2)[-2:];margins.append(dict(row=int(row),baseline_top1=int(top_gold[row]),candidate_top1=int(top_test[row]),baseline_top2_margin=float(two.max()-two.min()),baseline_score_of_candidate_top1=float(gold[row,top_test[row]]),baseline_top1_score=float(gold[row,top_gold[row]])))
      steps.append(dict(step=step,relative_RMS=rms,max_abs=float(np.max(np.abs(diff))),bitwise_logits_equal=bool(np.array_equal(gold,test)),top1_mismatches=len(mismatch),rows=len(gold),mismatch_margins=margins))
     comparison.append(dict(history_chunk=chunk,round=rid,mix=gold_case['mix'],steps=steps))
  result=dict(records=records,comparison=comparison,files=files,source_binary_SHA256=sha(binary),validation_source_SHA256=sha(package/'roofline_prefill_validation.cpp'),input_provenance=provenance,same_PTD_bytes=True,teacher_forced_tokens_identical=True,performance_and_F1_not_measured=True,rounds=4,no_numerical_acceptance_threshold_invented=True)
  (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');state('completed-pending-independent-numerical-review')
 except BaseException as e:state('failed',error=str(e));raise
if __name__=='__main__':main()
