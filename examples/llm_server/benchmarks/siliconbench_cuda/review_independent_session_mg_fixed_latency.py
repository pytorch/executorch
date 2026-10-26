"""Reconstruct teacher-forced inputs, actual methods, raw timings and mini logits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

work=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.job;output=args.output
assert not output.exists()
document=json.loads((root/'result.json').read_text())
assert document['fixed_teacher_forced_inputs']
if document.get('full_model_profile'):
 from independent_session_mini_profiles import validate_full_audit
 assert document['full_model_profile']=='muse-glimmer-52layers' and not document['mini_only']
 assert document['full_model_qualification_control']
 for arm,audit in document['artifact_audits'].items():
  assert validate_full_audit(audit,'muse-glimmer-52layers',arm=='independent')==document['model_geometry']
else:
 assert document['mini_only']
assert len(document['rounds'])==8
cases=[('decode'+str(c),[1]*c) for c in (1,2,4,8,16)]
cases += [('prefill64',[64]),('prefill512',[512]),('decode8-prefill64',[1]*8+[64]),('decode8-prefill512',[1]*8+[512])]
records=[]
logits={}
requests={}
raw_hashes={}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def token(owner,position):return 100+(owner*101+position*37)%149000
def counters(snapshot):return {item['method']:item['calls'] for item in snapshot['method_calls']}
def moments(values):return {'mean':statistics.mean(values),'sample_SD':statistics.stdev(values),'values':values}
for row in document['rounds']:
 number,arm=row['round'],row['arm']
 directory=root/f'round-{number}'/arm
 path=directory/'protocol.jsonl'
 assert digest(path)==row['protocol_sha256']
 entries=[json.loads(line) for line in path.read_text().splitlines()]
 operations=[]
 pending=None
 ready=[]
 for entry in entries:
  if 'request' in entry:
   assert pending is None
   pending=entry['request']
  else:
   reply=entry['response']
   if reply.get('ready'):
    assert pending is None
    ready.append(reply)
   else:
    assert pending is not None and reply['op']==pending['op'] and not reply.get('failed')
    assert math.isfinite(reply['elapsed_seconds']) and reply['elapsed_seconds']>=0
    operations.append((pending,reply));pending=None
 assert pending is None and len(ready)==1
 cursor=0
 clean=[]
 def consume(expected):
  global cursor
  body,reply=operations[cursor];cursor+=1
  clean_body={key:value for key,value in body.items() if key!='tag'}
  assert clean_body==expected,(clean_body,expected)
  assert reply['snapshot']['pid']==ready[0]['snapshot']['pid']
  clean.append(clean_body)
  return reply
 measured=[]
 assert len(row['cases'])==9
 for index,(label,widths) in enumerate(cases):
  owners=[consume({'op':'new'})['seq_id'] for _ in widths]
  ids=[owner for owner in owners for _ in range(1024)]
  values=[token(i,p) for i in range(len(owners)) for p in range(1024)]
  for offset in range(0,len(ids),2048):
   size=min(2048,len(ids)-offset)
   consume({'op':'step','seq_ids':ids[offset:offset+size],'tokens':values[offset:offset+size],'select':[size-1],'argmax':True})
  before=consume({'op':'metrics'})['snapshot']
  positions=[1024]*len(owners)
  original=row['cases'][index]
  assert original['case']==label and original['widths']==widths and original['history_per_owner']==1024
  assert len(original['calls'])==16
  calls=[]
  expected_method='forward_'+str(next((n for n in (1,2,4,8,16,32,64) if n>=sum(widths)), 'others'))
  for cycle in range(16):
   ids,values,select=[],[],[]
   for i,(owner,width) in enumerate(zip(owners,widths)):
    ids.extend([owner]*width)
    values.extend(token(i,p) for p in range(positions[i],positions[i]+width))
    select.append(len(values)-1);positions[i]+=width
   reply=consume({'op':'step','seq_ids':ids,'tokens':values,'select':select,'argmax':True,'save_logits':number==1 and cycle==15})
   after=reply['snapshot']
   changed=any(after[key]!=before[key] for key in ('live_KV_bytes','row_growth_count','column_resize_count'))
   recorded=original['calls'][cycle]
   assert recorded['before']==before and recorded['after']==after
   assert recorded['native_seconds']==reply['elapsed_seconds'] and recorded['storage_changed']==changed
   assert recorded['selected_argmax']==reply['argmax'] and recorded['logits']==reply['logits']
   assert len(reply['argmax'])==len(widths)
   left,right=counters(before),counters(after)
   difference={name:right[name]-left[name] for name in right if right[name]!=left[name]}
   assert difference=={expected_method:1},(label,difference)
   calls.append({'cycle':cycle,'seconds':reply['elapsed_seconds'],'storage_changed':changed,
                 'actual_method':expected_method,'actual_model_forwards':1,'selected_argmax':reply['argmax']})
   if reply['logits']:
    assert number==1 and cycle==15 and len(reply['logits'])==1
    part=reply['logits'][0]
    local=directory/Path(part['file']).name
    assert part['shape']==[len(widths),202048] and local.stat().st_size==len(widths)*202048*4
    value=np.fromfile(local,dtype='<f4').reshape(part['shape'])
    assert np.isfinite(value).all()
    logits[(arm,label)]=value
    raw_hashes[str(local.relative_to(root))]=digest(local)
   before=after
  stable=[call['seconds'] for call in calls[4:] if not call['storage_changed']]
  assert stable
  summary={'cold_first_four_seconds':sum(call['seconds'] for call in calls[:4]),
           'mean_all_measured_native_ms':1000*statistics.mean(call['seconds'] for call in calls[4:]),
           'mean_stable_native_ms':1000*statistics.mean(stable)}
  assert all(summary[key]==original[key] for key in summary)
  assert len(stable)==original['stable_calls']
  measured.append({'case':label,'widths':widths,'history_per_owner':1024,'calls':calls,'stable_calls':len(stable),**summary})
  for owner in owners:consume({'op':'remove','seq_id':owner})
 assert cursor==len(operations)
 requests[(number,arm)]=clean
 records.append({'round':number,'arm':arm,'cases':measured,'protocol_SHA256':digest(path)})
for number in range(1,5):
 assert requests[(number,'global')]==requests[(number,'independent')]
 assert [{k:v for k,v in r.items() if k!='save_logits'} for r in requests[(number,'global')]]==[{k:v for k,v in r.items() if k!='save_logits'} for r in requests[(1,'global')]]
aggregate=[]
for index,(label,widths) in enumerate(cases):
 fields={}
 for key in ('cold_first_four_seconds','mean_all_measured_native_ms','mean_stable_native_ms'):
  arms={arm:moments([row['cases'][index][key] for row in sorted((r for r in records if r['arm']==arm),key=lambda r:r['round'])[1:]]) for arm in ('global','independent')}
  for arm in arms:
   published=next(row['statistics'][key] for row in document['aggregates'] if row['arm']==arm and row['case']==label)
   assert published['mean']==arms[arm]['mean'] and published['sample_SD']==arms[arm]['sample_SD']
  a,b=arms['global'],arms['independent']
  fields[key]={'arms':arms,'delta_percent':100*(b['mean']/a['mean']-1),
               'combined_3SE':3*math.sqrt((a['sample_SD']**2+b['sample_SD']**2)/3)}
 lv,rv=logits[('global',label)],logits[('independent',label)]
 delta=rv.astype(np.float64)-lv.astype(np.float64)
 norms=np.linalg.norm(lv.astype(np.float64),axis=1)
 numerics={'max_abs_error':float(np.abs(delta).max()),'max_relative_L2':float((np.linalg.norm(delta,axis=1)/np.maximum(norms,1e-30)).max()),
           'greedy_disagreements':int(np.count_nonzero(lv.argmax(axis=1)!=rv.argmax(axis=1))),
           'strict_logit_tolerance_pass':bool(np.allclose(lv,rv,rtol=1e-5,atol=1e-4))}
 aggregate.append({'case':label,'widths':widths,**fields,'numerics':numerics})
result={'raw_inputs_protocols_timings_and_actual_routes_verified':True,'all_nine_cases_single_mixed_forward_verified':True,
        'same_teacher_forced_tokens_between_all_arms_and_rounds':True,'rounds':records,'aggregates':aggregate,
        'saved_logit_files_SHA256':raw_hashes,'source_result_SHA256':digest(root/'result.json'),
        'model_performance_qualified':False,'candidate_promoted':False,
        'limitations':'Mini diagnostic with independently tuned exported GEMM configurations. Actual native intervals include host preparation, argmax and CUDA completion but exclude snapshot/IPC; stable-only and all-inclusive timings are separate. Compiler/math matching remains required.'}
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'raw_route_verification':True,'case_count':9,'processes':8,
                  'summary':[{'case':row['case'],'stable_delta_percent':row['mean_stable_native_ms']['delta_percent'],'numerics':row['numerics']} for row in aggregate]}))
