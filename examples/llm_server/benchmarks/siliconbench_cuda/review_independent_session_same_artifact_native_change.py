"""Compare actual recorded inputs and tensors for identical PTEs across SDKs."""
from pathlib import Path
import datetime,hashlib,json,statistics
import numpy as np
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before-job',type=Path,required=True)
p.add_argument('--after-job',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();roots=[a.before_job,a.after_job];out=a.output
assert not out.exists()
docs=[json.loads((r/'result.json').read_text()) for r in roots]
assert all(d['native_sampler_mode']=='off' for d in docs)
assert all(d.get('performance_rounds_unprofiled',True) for d in docs)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ops(p):
 operations=[];pending=None;ready=[]
 for line in p.read_text().splitlines():
  e=json.loads(line)
  if 'request' in e:assert pending is None;pending=e['request']
  elif e['response'].get('ready'):assert pending is None;ready.append(e['response'])
  else:
   assert pending is not None and e['response']['op']==pending['op'] and not e['response'].get('failed')
   operations.append((pending,e['response']));pending=None
 assert pending is None and len(ready)==1
 return operations
def moments(v):return {'mean':statistics.mean(v),'sample_SD':statistics.stdev(v),'values':v}
records=[];full=[];argmax_count=0
for number in range(1,5):
 for arm in ('global','independent'):
  pairs=[next(row for row in d['rounds'] if row['round']==number and row['arm']==arm) for d in docs]
  paths=[root/f'round-{number}'/arm/'protocol.jsonl' for root in roots]
  for row,path in zip(pairs,paths):assert digest(path)==row['protocol_sha256']
  operations=[ops(path) for path in paths]
  assert len(operations[0])==len(operations[1])
  checked=0
  for left,right in zip(*operations):
   requests=[{k:v for k,v in item[0].items() if k!='tag'} for item in (left,right)]
   assert requests[0]==requests[1]
   if requests[0]['op']=='step':
    assert left[1]['argmax']==right[1]['argmax']
    checked+=len(left[1]['argmax'])
  argmax_count+=checked
  assert len(pairs[0]['cases'])==len(pairs[1]['cases'])==9
  for old,new in zip(pairs[0]['cases'],pairs[1]['cases']):
   assert old['case']==new['case'] and old['widths']==new['widths']
   assert len(old['calls'])==len(new['calls'])==16
   if number==1:
    for lc,rc in zip(old['calls'],new['calls']):
     assert len(lc['logits'])==len(rc['logits'])
     for lrecord,rrecord in zip(lc['logits'],rc['logits']):
      assert lrecord['shape']==rrecord['shape']
      raw=[root/f'round-{number}'/arm/Path(record['file']).name for root,record in zip(roots,(lrecord,rrecord))]
      sha=[digest(p) for p in raw];assert sha[0]==sha[1],(arm,old['case'])
      values=[np.fromfile(p,dtype='<f4') for p in raw]
      assert np.isfinite(values[0]).all() and np.array_equal(*values)
      full.append({'arm':arm,'case':old['case'],'shape':lrecord['shape'],'both_SHA256':sha[0],
                   'old_file':str(raw[0]),'new_file':str(raw[1]),'bitwise_equal':True})
  records.append({'round':number,'arm':arm,'selected_argmax_values_equal':checked,
                  'protocol_SHA256':[digest(p) for p in paths]})
artifacts=[]
for arm in ('global','independent'):
 audits=[d['artifact_audits'][arm] for d in docs]
 assert audits[0]['artifacts']==audits[1]['artifacts']
 artifacts.append({'arm':arm,'artifacts':audits[0]['artifacts']})
assert len(full)==18
cells=[]
for arm in ('global','independent'):
 for case in docs[0]['rounds'][0]['cases']:
  fields={}
  for name in ('cold_first_four_seconds','mean_stable_native_ms'):
   values=[moments([next(c for c in r['cases'] if c['case']==case['case'])[name] for r in d['rounds'] if r['arm']==arm and r['round']>=2]) for d in docs]
   delta=values[1]['mean']-values[0]['mean'];bound=3*((values[0]['sample_SD']**2+values[1]['sample_SD']**2)/3)**0.5
   fields[name]={'before':values[0],'after':values[1],'delta':delta,'delta_percent':100*(values[1]['mean']/values[0]['mean']-1),'combined_3SE':bound,'positive_signal':delta>bound}
  cells.append({'arm':arm,'case':case['case'],**fields})
result={'cells':cells,'same_compiled_artifacts':True,'both_native_sampler_OFF':True,'no_model_TPS_or_F1_qualification':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'same_PTE_and_PTD_recorded_hashes_verified':True,'all_four_round_requests_and_argmax_verified':True,
        'selected_argmax_values_equal':argmax_count,'full_saved_FP32_tensors_equal':len(full),
        'native_change_preserves_observed_model_numerics':True,
        'records':records,'full_tensors':full,'artifacts':artifacts,
        'native_gate_SHA256':[d['native_gate_sha256'] for d in docs],
        'probe_SHA256':[d['probe_sha256'] for d in docs],
        'original_results_SHA256':[digest(root/'result.json') for root in roots],
        'limitations':'Four-round fixed teacher-forced mini control, identical compiled PTE/PTD per arm; native SDK versions differ. Saved full logits in round 1 only. Round1 may include the explicitly recorded trace; all averaged performance rounds2-4 are unprofiled. Native model intervals include host work, argmax and CUDA completion; exclude snapshot/IPC. Native-change timing signals are reported without asserting zero overhead. Full-model and HTTP/F1 qualification remain separate.'}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'same_PTE_and_PTD':True,'argmax_values_equal':argmax_count,'full_saved_tensors_bitwise_equal':len(full)}))
