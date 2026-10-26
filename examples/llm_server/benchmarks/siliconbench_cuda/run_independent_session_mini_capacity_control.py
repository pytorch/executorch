"""Compare runtime allocation with the exact same independent mini artifact.

One arm starts full-attention owners at512 rows and grows independently.
The reference starts them at16384 rows, covering every tested history without
row growth. The sliding policy, compiled math, native SDK, token operations and
automatic physical-column policy are identical. This numerical allocation
control complements the separate global-layout export and graph/perf probes.
It does not select a main preset or qualify the full models by itself.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from independent_session_mini_profiles import DEFAULT_PROFILE,PROFILES,validate_mini_audit

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('preparation-job','output','lock'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mini-profile',choices=PROFILES,default=DEFAULT_PROFILE)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 def state(status,**extra):
  temp=a.output/'progress.pending.json'
  temp.write_text(json.dumps({'status':status,'controller_pid':os.getpid(),
   'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_model_qualified':False,
   'candidate_accepted':False,**extra},indent=2)+'\n');temp.replace(a.output/'progress.json')
 try:
  prep=json.loads((a.preparation_job/'progress.json').read_text());assert prep['status']=='mini-artifacts-ready'
  gate_path=a.preparation_job/'native-gate.json';gate=json.loads(gate_path.read_text())
  assert gate['native_GPU_tests_pass'] and gate['test_count']==74
  artifact=a.preparation_job/'mini-independent-artifact'
  audit=json.loads((artifact/'independent-mini-artifact-audit.json').read_text())
  validate_mini_audit(audit,a.mini_profile,True)
  assert audit['source_spec_sha256']==gate['source_spec_SHA256']
  probe=a.preparation_job/'model-build/muse_glimmer_runtime_kv_probe'
  assert digest(probe)==gate['binaries'][str(probe)]
  package=Path(__file__).resolve().parent
  roots={};documents={}
  for name,capacity in (('preallocated16384',16384),('growing512',512)):
   root=a.output/name;roots[name]=root
   command=[sys.executable,'-B','-u',str(package/'run_independent_session_mini_probe.py'),
    '--probe',str(probe),'--artifact',str(artifact),'--native-gate',str(gate_path),
    '--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
    '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
    '--arm','independent','--mini-profile',a.mini_profile,'--automatic-reclaim',
    '--job-dir',str(root),'--lock',str(a.lock),'--diagnostic-initial-KV-capacity',str(capacity)]
   state('running-same-PTE-capacity-control',arm=name,initial_capacity=capacity)
   with (a.output/(name+'-driver.log')).open('x') as log:
    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=7200,
      env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'})
   doc=json.loads((root/'result.json').read_text());documents[name]=doc
   assert doc['initial_KV_capacity']==capacity and len(doc['rounds'])==4
  import numpy as np
  rounds=[];files={}
  def operations(root,number,expected):
   path=root/f'round-{number}/protocol.jsonl';assert digest(path)==expected
   result=[];pending=None
   for line in path.read_text().splitlines():
    value=json.loads(line)
    if 'request' in value:assert pending is None;pending=value['request']
    elif value['response'].get('ready'):assert pending is None
    else:
     assert pending is not None and pending['op']==value['response']['op']
     assert not value['response'].get('failed')
     result.append((pending,value['response']));pending=None
   assert pending is None
   return result
  checks=[]
  for number in range(1,5):
   data=[operations(roots[name],number,documents[name]['rounds'][number-1]['protocol_sha256'])
         for name in roots]
   assert len(data[0])==len(data[1]);steps=0;argmax_differences=[]
   for i,((lr,l),(rr,r)) in enumerate(zip(*data)):
    assert {k:v for k,v in lr.items() if k!='tag'}=={k:v for k,v in rr.items() if k!='tag'}
    if lr['op']!='step':continue
    steps+=1
    if l['argmax']!=r['argmax']:argmax_differences.append(i)
    assert len(l['logits'])==len(r['logits'])
    for lp,rp in zip(l['logits'],r['logits']):
     assert lp['shape']==rp['shape'] and lp['selected_indices']==rp['selected_indices']
     arrays=[]
     for part,root in zip((lp,rp),roots.values()):
      path=Path(part['file']);assert path.resolve().is_relative_to(root.resolve())
      arrays.append(np.fromfile(path,dtype='<f4').reshape(part['shape']))
      files[str(path)]=digest(path)
     assert all(np.isfinite(x).all() for x in arrays)
     checks.append({'round':number,'operation':i,'shape':lp['shape'],
      'bitwise_equal':bool(np.array_equal(*arrays)),
      'max_abs':float(np.abs(arrays[0].astype(np.float64)-arrays[1]).max())})
   rounds.append({'round':number,'step_count':steps,'argmax_difference_operations':argmax_differences})
  assert all(doc['model_PTE_sha256']==documents['growing512']['model_PTE_sha256'] and
             doc['weights_blob_sha256']==documents['growing512']['weights_blob_sha256'] and
             doc['probe_sha256']==documents['growing512']['probe_sha256'] for doc in documents.values())
  passed=not any(row['argmax_difference_operations'] for row in rounds) and all(row['bitwise_equal'] for row in checks)
  result={'same_PTE_PTD_probe_and_native_SDK':True,'same_token_operations':True,
    'mini_profile':a.mini_profile,'reference_initial_capacity':16384,'candidate_initial_capacity':512,
    'rounds':rounds,'full_logit_checks':checks,'saved_logits_SHA256':files,
    'bitwise_allocation_equivalence_pass':passed,'compiled_math_changed':False,
    'sliding_policy_changed':False,'full_model_qualified':False,'candidate_accepted':False,
    'source_result_SHA256':{name:digest(root/'result.json') for name,root in roots.items()},
    'PTE_SHA256':documents['growing512']['model_PTE_sha256'],
    'PTD_SHA256':documents['growing512']['weights_blob_sha256']}
  (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  state('same-PTE-capacity-control-collected-pending-independent-review',bitwise_equivalence_pass=passed)
 except Exception as error:
  state('failed',detail=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
