"""Attribute saved native calls to kernels, graph capture and allocator APIs.

Read only the discarded profiled round. Hash actual child results, traces and
protocols. Preserve timing gates; sums can overlap and are not critical paths.
"""
import argparse,collections,datetime,json,sqlite3,statistics
from pathlib import Path
from independent_session_full_native_binding import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--matched-job',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();assert not a.output.exists()
 matched=json.loads((a.matched_job/'result.json').read_text())
 assert matched['four_rounds_discard_first'] and matched['ABBA_native_order']
 assert not matched['main_candidate_accepted']
 out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  matched_result_SHA256=sha(a.matched_job/'result.json'),profiled_discarded_round_only=True,
  original_positive_signals=matched['positive_performance_signals'],automatic_performance_waiver=False,
  sums_may_overlap_not_additive_critical_path=True,arms={})
 for side in ('before','after'):
  child=a.matched_job/f'child-round-1-{side}';record=next(x for x in matched['actual_child_results'] if x['native']==side and x['round']==1)
  assert Path(record['result']).resolve()==(child/'result.json').resolve()
  assert sha(child/'result.json')==record['SHA256']
  doc=json.loads((child/'result.json').read_text());all_cases=[]
  for row in doc['rounds']:
   assert row['round']==1 and row['profiled_discarded_round']
   root=child/'round-1'/row['arm'];protocol=root/'protocol.jsonl';assert sha(protocol)==row['protocol_sha256']
   database=root/'cold-profile.sqlite';db=sqlite3.connect('file:'+str(database.resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
   strings=dict(db.execute('select id,value from StringIds'))
   ranges={x['text'] or strings.get(x['textId']):dict(x) for x in db.execute('select * from NVTX_EVENTS') if (x['text'] or strings.get(x['textId']) or '').startswith('latency-')}
   kernels=[{**dict(x),'name':strings.get(x['shortName'],str(x['shortName']))} for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
   runtime=[{**dict(x),'name':strings[x['nameId']]} for x in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
   pairs={};pending=None
   for line in protocol.read_text().splitlines():
    event=json.loads(line)
    if 'request' in event:assert pending is None;pending=event['request']
    elif pending is not None:
     assert event['response']['op']==pending['op'] and not event['response'].get('failed')
     assert pending['tag'] not in pairs;pairs[pending['tag']]=(pending,event['response']);pending=None
   assert pending is None
   used=set()
   for case in row['cases']:
    calls=[]
    for call in case['calls']:
     matches=[(tag,req,res) for tag,(req,res) in pairs.items() if tag not in used and req['op']=='step' and res['elapsed_seconds']==call['native_seconds'] and res['argmax']==call['selected_argmax'] and len(req['tokens'])==sum(case['widths'])]
     assert len(matches)==1,(side,row['arm'],case['case'],call['cycle'])
     tag,req,res=matches[0];used.add(tag);interval=ranges[tag]
     active=[x for x in kernels if interval['start']<=x['start'] and x['end']<=interval['end']]
     api=sorted([x for x in runtime if x['globalTid']==interval['globalTid'] and interval['start']<=x['start'] and x['end']<=interval['end']],key=lambda x:x['start'])
     capture=[];start=None
     for event in api:
      if 'StreamBeginCapture' in event['name']:assert start is None;start=event['start']
      elif 'StreamEndCapture' in event['name']:assert start is not None;capture.append((start,event['end']));start=None
     assert start is None
     kernel_ms=collections.defaultdict(float)
     for x in active:kernel_ms[x['name']]+=(x['end']-x['start'])/1e6
     inst=[x for x in api if 'GraphInstantiate' in x['name']]
     allocator=[x for x in api if any(k in x['name'] for k in ('Malloc','Free','MemPoolTrim'))]
     calls.append(dict(cycle=call['cycle'],native_ms=1000*call['native_seconds'],storage_changed=call['storage_changed'],method=call['actual_method'],actual_model_forwards=call['actual_model_forwards'],kernel_ms=dict(kernel_ms),kernel_sum_ms=sum(kernel_ms.values()),capture_count=len(capture),capture_span_ms=sum(e-s for s,e in capture)/1e6,instantiate_count=len(inst),instantiate_API_ms=sum(x['end']-x['start'] for x in inst)/1e6,allocator_API_ms=sum(x['end']-x['start'] for x in allocator)/1e6,API_counts=dict(collections.Counter(x['name'] for x in api))))
    stable=[x for x in calls if x['cycle']>=4 and not x['storage_changed']]
    assert len(stable)==case['stable_calls'] and all(x['actual_model_forwards']==1 for x in calls)
    all_cases.append(dict(layout=row['arm'],case=case['case'],SQLite_SHA256=sha(database),protocol_SHA256=sha(protocol),calls=calls,stable_means={k:statistics.mean(x[k] for x in stable) for k in ('native_ms','kernel_sum_ms','capture_count','allocator_API_ms')},cold_sums={k:sum(x[k] for x in calls[:4]) for k in ('native_ms','capture_count','capture_span_ms','instantiate_count','instantiate_API_ms','allocator_API_ms')}))
   db.close()
  out['arms'][side]=all_cases
 assert [(x['layout'],x['case']) for x in out['arms']['before']]==[(x['layout'],x['case']) for x in out['arms']['after']]
 out['comparisons']=[dict(layout=b['layout'],case=b['case'],stable_delta={k:c['stable_means'][k]-b['stable_means'][k] for k in b['stable_means']},cold_delta={k:c['cold_sums'][k]-b['cold_sums'][k] for k in b['cold_sums']}) for b,c in zip(out['arms']['before'],out['arms']['after'])]
 with a.output.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(raw_trace_review_complete=True,comparisons=out['comparisons'],original_signals_preserved=True)))

if __name__=='__main__':main()
