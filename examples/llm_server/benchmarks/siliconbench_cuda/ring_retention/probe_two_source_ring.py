# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Standalone prerequisite. Does not claim to qualify an ET mini/full model."""
import argparse,hashlib,importlib.util,json,math,statistics,time,traceback
from pathlib import Path
import torch
from two_source_ring import sdpa_ring_current,ring_commit_current

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',choices=['MG','Gemma'],required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
assert torch.cuda.is_available()
torch.manual_seed(49027)
profile=dict(window=2048,dim=128,kvheads=2,qheads=32,budget=4096) if a.model=='MG' else dict(window=1024,dim=256,kvheads=8,qheads=16,budget=2048)
W,D,HK,HQ,P=(profile[k] for k in ['window','dim','kvheads','qheads','budget'])
SCALE=43.7840518911/D if a.model=='MG' else 1/math.sqrt(D)
profile['scale']=SCALE
spec=importlib.util.spec_from_file_location('current_et_baseline',a.baseline)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
results=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(status,**extra):
 v=dict(status=status,model=a.model,profile=profile,kernel_source_SHA256=sha(Path(__file__).with_name('two_source_ring.py')),baseline_source_SHA256=sha(a.baseline),results=results,**extra)
 (a.output/'progress.json').write_text(json.dumps(v,indent=2)+'\n')


def fixture(lengths,history_lengths,pad=0):
 columns=len(lengths)+1
 total=sum(lengths)+pad
 q=torch.randn(1,HQ,total,D,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(1,HK,total,D,device='cuda',dtype=torch.bfloat16)
 v=torch.randn_like(k)
 smallk=torch.zeros(W,columns,HK,D,device='cuda',dtype=torch.bfloat16)
 smallv=torch.zeros_like(smallk)
 oldr=W+P-1
 bigk=torch.zeros(oldr,columns,HK,D,device='cuda',dtype=torch.bfloat16)
 bigv=torch.zeros_like(bigk)
 info=torch.zeros(len(lengths)+2,7,device='cuda',dtype=torch.int32)
 rows=[];history=[];start=0
 for col,(n,h) in enumerate(zip(lengths,history_lengths)):
  assert n<=P and h>=0
  kh=torch.randn(h,HK,D,device='cuda',dtype=torch.bfloat16)
  vh=torch.randn_like(kh)
  history.append((kh,vh))
  if h:
   tail=torch.arange(max(h-W,0),h,device='cuda');smallk[tail%W,col]=kh[tail];smallv[tail%W,col]=vh[tail]
   tail=torch.arange(max(h-oldr,0),h,device='cuda');bigk[tail%oldr,col]=kh[tail];bigv[tail%oldr,col]=vh[tail]
  rows.append([start,n,col,h+n,columns,0,0]);start+=n
 info[:len(rows)]=torch.tensor(rows,device='cuda',dtype=torch.int32)
 return dict(initial={'sk':smallk.clone(),'sv':smallv.clone(),'bk':bigk.clone(),'bv':bigv.clone()},q=q,k=k,v=v,sk=smallk,sv=smallv,bk=bigk,bv=bigv,info=info,rows=rows,histories=history,lengths=lengths,history_lengths=history_lengths,pad=pad)


def oracle(f):
 out=torch.zeros_like(f['q']);start=0
 for n,h,(kh,vh) in zip(f['lengths'],f['history_lengths'],f['histories']):
  keep=min(h,W-1);first=h-keep
  ka=torch.cat((kh[-keep:] if keep else kh[:0],f['k'][0,:,start:start+n].transpose(0,1)),0)
  va=torch.cat((vh[-keep:] if keep else vh[:0],f['v'][0,:,start:start+n].transpose(0,1)),0)
  for j in range(0,n,16):
   count=min(16,n-j)
   query=f['q'][0,:,start+j:start+j+count].float()
   keys=ka.transpose(0,1).repeat_interleave(HQ//HK,0).float()
   vals=va.transpose(0,1).repeat_interleave(HQ//HK,0).float()
   score=torch.matmul(query,keys.transpose(1,2))*SCALE
   kp=torch.arange(first,h+n,device='cuda');qp=torch.arange(h+j,h+j+count,device='cuda')
   visible=(kp[None,:]<=qp[:,None])&(kp[None,:]>qp[:,None]-W)
   score=score.masked_fill(~visible[None],-float('inf'))
   probs=score.softmax(-1)
   out[0,:,start+j:start+j+count]=torch.matmul(probs,vals).to(torch.bfloat16)
  start+=n
 return out


def old_write(f):
 # Same scatter kernel, with large original ring capacity, commits all rows.
 ring_commit_current(f['bk'],f['bv'],f['k'],f['v'],f['info'],W+P-1)


def baseline(f):
 old_write(f)
 return mod.sdpa_varlen(f['q'],f['bk'],f['bv'],f['info'],1*SCALE,True,W,W+P-1)


def candidate(f,commit=True):
 out=sdpa_ring_current(f['q'],f['sk'],f['sv'],f['k'],f['v'],f['info'],W,1*SCALE)
 if commit:ring_commit_current(f['sk'],f['sv'],f['k'],f['v'],f['info'],W)
 return out


def errors(x,y,total):
 d=(x[:,:,:total].float()-y[:,:,:total].float()).flatten()
 ref=y[:,:,:total].float().flatten()
 return dict(max_abs=float(d.abs().max()),normalized_RMS=float(d.square().mean().sqrt()/(ref.square().mean().sqrt()+1e-12)))


def check_commit(f):
 for start,n,col,end,_,_,_ in f['rows']:
  keep=min(n,W);ts=torch.arange(start+n-keep,start+n,device='cuda');pos=torch.arange(end-keep,end,device='cuda')%W
  assert torch.equal(f['sk'][pos,col],f['k'][0,:,ts].transpose(0,1))
  assert torch.equal(f['sv'][pos,col],f['v'][0,:,ts].transpose(0,1))
 # Static padding has no segment; dummy-column storage must stay untouched.
 assert torch.count_nonzero(f['sk'][:,-1]).item()==0
 assert torch.count_nonzero(f['sv'][:,-1]).item()==0


def timing(f):
 # Captured event nodes exclude Python/JIT dispatch and state restore from
 # the measured device-forward interval. History restore stays in the graph
 # before its timing event so each replay is the same valid forward.
 saved=f['initial']
 graphs={};events={};samples={x:[] for x in ['baseline','candidate']}
 for name,fn,keys in [('baseline',baseline,['bk','bv']),('candidate',candidate,['sk','sv'])]:
  for _ in range(3):
   for key in keys:f[key].copy_(saved[key])
   fn(f)
  torch.cuda.synchronize()
  g=torch.cuda.CUDAGraph()
  begin=torch.cuda.Event(enable_timing=True,external=True)
  finish=torch.cuda.Event(enable_timing=True,external=True)
  with torch.cuda.graph(g):
   for key in keys:f[key].copy_(saved[key])
   begin.record();output=fn(f);finish.record()
  graphs[name]=g;events[name]=(begin,finish,output)
 for rnd in range(4):
  for name in ['baseline','candidate']:
   vals=[]
   for _ in range(12):
    graphs[name].replay();torch.cuda.synchronize()
    vals.append(events[name][0].elapsed_time(events[name][1]))
   samples[name].append(statistics.mean(vals[2:]))
 return {name:dict(round_means_ms=vals,discarded_round=1,retained_mean_ms=statistics.mean(vals[1:]),sample_SD_ms=statistics.stdev(vals[1:]),CUDA_graph=True,state_restore_excluded=True,external_event_nodes=True) for name,vals in samples.items()}

cases=[('empty-d1',[1],[0],0),('history-d1',[1],[W*3+7],0),('d8',[1]*8,[W*2+i*13 for i in range(8)],0),('d16',[1]*16,[W*3+i*31 for i in range(16)],0),('p64',[64],[W*2+37],0),('p512',[512],[W*3+11],0),('max-prefill',[P],[W*2+17],0),('mixed64',[1]*8+[64],[W*3+i*11 for i in range(9)],0),('mixed512',[1]*8+[512],[W*2+i*7 for i in range(9)],0),('unequal-tail-pad',[1,3,17],[0,W-1,W+7],43)]
try:
 record('running-kernel-prerequisite')
 for name,lengths,hist,pad in cases:
  assert sum(lengths)<=P
  f=fixture(lengths,hist,pad);ref=oracle(f)
  old=baseline(f);new=candidate(f);torch.cuda.synchronize()
  oe=errors(old,ref,sum(lengths));ne=errors(new,ref,sum(lengths));delta=errors(new,old,sum(lengths))
  assert ne['normalized_RMS']<=max(.01,3*oe['normalized_RMS']),dict(case=name,candidate=ne,baseline=oe)
  check_commit(f)
  t=timing(f)
  results.append(dict(case=name,lengths=lengths,history_lengths=hist,padding=pad,baseline_vs_FP32_oracle=oe,candidate_vs_FP32_oracle=ne,candidate_vs_baseline=delta,commit_exact=True,padding_writes_isolated=True,attention_and_commit_timing=t,history_bytes_baseline=f['bk'].numel()*f['bk'].element_size()*2,history_bytes_candidate=f['sk'].numel()*f['sk'].element_size()*2))
  record('running-kernel-prerequisite',last_completed=name)
  print(json.dumps({'case':name,'candidate_RMS':ne['normalized_RMS'],'baseline_ms':t['baseline']['retained_mean_ms'],'candidate_ms':t['candidate']['retained_mean_ms']}),flush=True)
  del f,ref,old,new
  torch.cuda.empty_cache()
 record('kernel-prerequisite-collected',kernel_numerical_gate_passed=True,ET_mini_model_qualified=False,allocator_policy_qualified=False,graph_dynamic_metadata_qualified=False,native_ring_storage_qualified=False,scope='Actual CUDA kernel prerequisites only; four-condition mini model/native qualification remains required.')
 (a.output/'result.json').write_text((a.output/'progress.json').read_text())
except BaseException as e:
 record('failed-kernel-prerequisite',error=repr(e),traceback=traceback.format_exc());raise
