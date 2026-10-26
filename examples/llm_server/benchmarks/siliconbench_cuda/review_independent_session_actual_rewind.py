"""Independently verify model lifecycle and actual rewind against saved raw tensors."""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np
w=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',choices=['gemma4-first6','muse-glimmer-first4'],required=True)
a=p.parse_args();root=a.job;vocab=262144 if a.profile=='gemma4-first6' else 202048
doc=json.loads((root/'result.json').read_text())
assert doc['same_artifact_same_forward_same_tokens'] and doc['actual_rewind_checked']
arms=doc.get('collected_arms',['global','independent']);assert arms in (['global','independent'],['global'],['independent'])
assert len(doc['rounds'])==4*len(arms) and doc['mini_profile']==a.profile
assert {(r['round'],r['arm']) for r in doc['rounds']}=={(i,arm) for i in range(1,5) for arm in arms}
verified=[];invalid_count=0;protocols={};artifacts={};owner_pids=set()
for row in doc['rounds']:
 rid,arm=row['round'],row['arm'];directory=root/f'round-{rid}'/arm
 path=directory/'protocol.jsonl';raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['protocol_SHA256']
 operations=[];pending=None;ready=[]
 for e in (json.loads(s) for s in raw.splitlines()):
  if 'request' in e:
   assert pending is None;pending=e['request']
  elif e['response'].get('ready'):
   assert pending is None;ready.append(e['response'])
  else:
   reply=e['response'];assert pending is not None and reply['op']==pending['op'] and not reply.get('failed')
   operations.append((pending,reply));pending=None
 assert pending is None and len(ready)==1
 owner_pids.add(ready[0]['snapshot']['pid'])
 saved={}
 previous_snapshot=ready[0]['snapshot']
 for req,reply in operations:
  if req['op']=='step' and req.get('save_logits'):
   assert len(reply['logits'])==1
   part=reply['logits'][0];assert part['shape']==[1,vocab]
   assert len(req['tokens'])==8 and len(set(req['seq_ids']))==1 and req['select']==[7]
   left={x['method']:x['calls'] for x in previous_snapshot['method_calls']};right={x['method']:x['calls'] for x in reply['snapshot']['method_calls']}
   assert {k:right[k]-left[k] for k in right if right[k]!=left[k]}=={'forward_8':1}
   saved[Path(part['file']).name]=(req,reply,part)
  if req['op']=='rewind' and req.get('expect_success') is False:
   assert req['position']==131073 and reply['rewound'] is False
   assert reply['position'] in (520,8200);invalid_count+=1
  previous_snapshot=reply['snapshot']
 expected={(h,p) for h in (512,8192) for p in ['clone','clone-actual-rewind','after-other-owner-growth','after-other-owner-growth-actual-rewind','after-owner-deletion','after-owner-deletion-actual-rewind']}
 assert {(c['history'],c['phase']) for c in row['checks']}==expected
 for c in row['checks']:
  arrays=[];requests=[]
  for name in ('reference','actual'):
   record=c[name];local=directory/Path(record['file']).name;raw=local.read_bytes();assert hashlib.sha256(raw).hexdigest()==record['SHA256']
   arr=np.frombuffer(raw,dtype='<f4');assert arr.size==vocab and np.isfinite(arr).all();arrays.append(arr)
   req,reply,part=saved[local.name];assert reply['snapshot']==record['snapshot'] and reply['argmax']==record['argmax']
   assert int(np.argmax(arr))==record['argmax'][0];requests.append(req)
  assert requests[0]['tokens']==requests[1]['tokens']==[100+(37*p)%149000 for p in range(c['history'],c['history']+8)]
  assert np.array_equal(*arrays) and c['bitwise_equal'] and c['max_abs']==0
  if c['phase'].endswith('-actual-rewind'):
   r=c['rewind_response'];assert r['op']=='rewind' and r['rewound'] and r['position']==c['history']
   assert r in [reply for req,reply in operations if req['op']=='rewind']
  verified.append({'round':rid,'arm':arm,'history':c['history'],'phase':c['phase'],'bitwise_logits_equal':True,'raw_reference_SHA256':c['reference']['SHA256'],'raw_actual_SHA256':c['actual']['SHA256']})
 artifacts.setdefault(arm,set()).add(row['PTE_SHA256'])
 protocols[rid,arm]=[{k:v for k,v in req.items() if k!='tag'} for req,reply in operations]
assert len(verified)==48*len(arms) and invalid_count==24*len(arms) and all(len(a)==1 for a in artifacts.values())
assert len(owner_pids)==4*len(arms)
out=a.output;assert not out.exists()
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_reads_local_CPU':True,'raw_logits_protocols_and_actual_rewind_verified':True,'checks':verified,'collected_arms':arms,'default_two_arm_collection':arms==['global','independent'],'invalid_out_of_context_rewinds_rejected':invalid_count,'same_PTE_within_each_arm_across_rounds':{k:next(iter(v)) for k,v in artifacts.items()},'cross_export_equivalence_not_claimed':True,'actual_graph_replay_review_separate':True,'model_performance_qualified':False,'full_model_promoted':False,'raw_directory':str(root),'source_result_SHA256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest()}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(verified),'invalid_rewinds':invalid_count,'collected_arms':arms,'report':str(out)}))
