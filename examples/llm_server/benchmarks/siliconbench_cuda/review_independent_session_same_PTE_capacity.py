"""Independently check preserved capacity-control raw tensors and native protocols."""
import argparse, datetime, hashlib, json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def counts(s): return {v['method']: v['calls'] for v in s['method_calls']}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job',type=Path,required=True)
p.add_argument('--remote-source',type=Path,help='Original absolute root when reviewing read-only local copies')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
root=a.job
receipt={'remote_source':str(a.remote_source or root.resolve())}
parent=load(root/'result.json')
arms=('preallocated16384','growing512')
docs={name:load(root/name/'result.json') for name in arms}
keys=('model_PTE_sha256','weights_blob_sha256','probe_sha256','native_gate_sha256','model_geometry','mini_profile')
assert all(docs[arms[0]][k]==docs[arms[1]][k] for k in keys)
for arm,capacity in zip(arms,(16384,512)):
    doc=docs[arm]
    assert doc['initial_KV_capacity']==capacity and doc['same_independent_PTE_initial_capacity_diagnostic']
    assert doc['automatic_reclaim_requested'] and doc['arm']=='independent'
    assert sha(root/arm/'result.json')==parent['source_result_SHA256'][arm]

def protocol(arm,n):
    path=root/arm/f'round-{n}'/'protocol.jsonl'
    assert sha(path)==docs[arm]['rounds'][n-1]['protocol_sha256']
    rows=[json.loads(v) for v in path.read_text().splitlines()]
    assert all(l['monotonic_ns']<=r['monotonic_ns'] for l,r in zip(rows,rows[1:]))
    result=[]; pending=None; before=None
    for row in rows:
        if 'request' in row:
            assert pending is None; pending=row['request']; continue
        response=row['response']; assert not response.get('failed')
        if response.get('ready'):
            assert pending is None and before is None; before=response['snapshot']; continue
        assert pending is not None and pending['op']==response['op']
        snapshot=response['snapshot']
        delta={k:v-counts(before)[k] for k,v in counts(snapshot).items() if v!=counts(before)[k]}
        if pending['op']=='step':
            assert len(pending['tokens'])==len(pending['seq_ids'])>0
            assert len(delta)==1 and list(delta.values())==[1], delta
            assert len(response['argmax'])==len(pending.get('select',[]))
        else: assert not delta
        result.append((pending,response)); pending=None; before=snapshot
    assert pending is None
    return result

rounds=[]; checks=[]; pids=set(); shapes=set()
for n in range(1,5):
    data=[protocol(arm,n) for arm in arms]
    assert len(data[0])==len(data[1])
    steps=0; argmax_matches=0
    for i,((lr,l),(rr,r)) in enumerate(zip(*data)):
        assert {k:v for k,v in lr.items() if k!='tag'}=={k:v for k,v in rr.items() if k!='tag'}
        if lr['op']!='step': continue
        steps+=1; assert l['argmax']==r['argmax']; argmax_matches+=1
        assert len(l['logits'])==len(r['logits'])
        for lp,rp in zip(l['logits'],r['logits']):
            assert lp['shape']==rp['shape'] and lp['selected_indices']==rp['selected_indices']
            assert lp['shape'][1]==docs[arms[0]]['model_geometry']['vocab']
            tensors=[]; hashes=[]
            for arm,part in zip(arms,(lp,rp)):
                remote=Path(part['file']); rel=remote.relative_to(Path(receipt['remote_source']))
                assert rel.parts[:2]==(arm,f'round-{n}')
                f=root/rel; assert sha(f)==parent['saved_logits_SHA256'][str(remote)]
                assert f.stat().st_size==np.prod(part['shape'])*4
                v=np.fromfile(f,dtype='<f4').reshape(part['shape']); assert np.isfinite(v).all()
                tensors.append(v); hashes.append(sha(f))
            # SHA equality checks bits, including sign-of-zero and all FP32 payloads.
            assert hashes[0]==hashes[1] and np.array_equal(*tensors)
            shapes.add(tuple(lp['shape']))
            checks.append(dict(round=n,operation=i,shape=lp['shape'],raw_SHA256=hashes[0],bitwise_equal=True))
    capacities={}; ops={}
    for arm,items in zip(arms,data):
        doc=docs[arm]['rounds'][n-1]
        snapshots=[v['snapshot'] for _,v in items]
        pidset={s['pid'] for s in snapshots}; assert len(pidset)==1 and not pids.intersection(pidset); pids.update(pidset)
        fullcaps=sorted({s['flat_capacity'] for s in snapshots if s['flat_capacity']})
        capacities[arm]=fullcaps
        if arm=='preallocated16384': assert fullcaps==[16384]
        else: assert fullcaps[0]==512 and fullcaps[-1]==16384 and len(fullcaps)>2
        assert [s['concurrency'] for s in doc['stages']]==[1,2,4,8,16,1]
        assert [s['after']['physical_columns'] for s in doc['stages']]==[2,3,5,9,17,2]
        assert doc['stages'][-1]['after']['live_sequences']==1
        assert not any(req['op']=='compact' for req,_ in items)
        pad=doc['post_shrink_method_checks']
        assert len(pad)==35
        assert sorted({x['static_width'] for x in pad})==[1,2,4,8,16,32,64]
        assert sorted({x['actual_tokens'] for x in pad})==[1,2,3,7,13,32,63]
        assert all(x['method_delta']=={'forward_'+str(x['static_width']):1} and
                   x['snapshot']['physical_columns']==2 and x['snapshot']['live_sequences']==1 for x in pad)
        ops[arm]={op:sum(req['op']==op for req,_ in items) for op in ('new','clone','remove','step','metrics')}
    assert ops[arms[0]]==ops[arms[1]]
    rounds.append(dict(round=n,steps=steps,all_step_argmax_equal=True,one_forward_per_step_verified=True,
                       full_capacity_regimes=capacities,operations=ops[arms[0]],
                       automatic_physical_columns=[2,3,5,9,17,2],post_shrink_padding_checks_per_arm=35))
assert len(checks)==len(parent['full_logit_checks']) and len(pids)==8
assert all(r['steps']==(489 if docs[arms[0]]['mini_profile']=='gemma4-first6' else 447) for r in rounds)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    raw_protocol_tensor_and_result_SHA256_verified=True,same_artifact_and_native_binary_hashes_verified=True,
    bitwise_allocation_equivalence_verified=True,original_model_tolerance_unchanged=True,
    full_model_profile=docs[arms[0]].get('full_model_profile'),
    model_geometry=docs[arms[0]]['model_geometry'],artifact_hashes={k:docs[arms[0]][k] for k in keys if 'sha256' in k},
    rounds=rounds,full_logit_checks=checks,full_logit_shapes=sorted(shapes),fresh_native_PIDs=sorted(pids),
    source_result_SHA256=sha(root/'result.json'),
    actual_graph_trace_not_in_this_control=True,full_model_qualified=False,candidate_promoted=False,
    limitations='Same compiled independent-layout PTE at two initial capacities; actual model geometry and full profile are recorded above. Full FP32 logits saved in round1 only; all four rounds verify every selected argmax and native one-forward routing. Aggregate full-capacity snapshots do not expose each owner; separate native tests prove per-owner allocation. Actual argv was not saved; effective runtime capacities and provenance are verified from raw native responses. Rewind and CUDA graph tracing are separate retained tests. This control is not cross-export numerical acceptance or task F1.')
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('bitwise_allocation_equivalence_verified','fresh_native_PIDs','full_model_qualified')}))
