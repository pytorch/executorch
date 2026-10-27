# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Compiled journal integration diagnostic; no mini or full model qualification claim."""
import argparse,hashlib,importlib.util,json,math,statistics,time,traceback
from pathlib import Path
import torch
from two_source_ring import sdpa_ring_current,ring_commit_current,ring_commit_journal

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',choices=['MG','Gemma'],required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--baseline-execution',choices=['eager','compiled'],default='eager')
p.add_argument('--candidate-execution',choices=['eager','compiled'],default='compiled')
p.add_argument('--generic-bm',type=int,default=0)
p.add_argument('--generic-bn',type=int,default=0)
p.add_argument('--generic-stages',type=int,default=0)
p.add_argument('--decode-max-tokens',type=int,default=0)
p.add_argument('--include-d8',action='store_true')
p.add_argument('--flat-history-bindings',action='store_true')
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
 v=dict(status=status,model=a.model,profile=profile,baseline_execution=a.baseline_execution,candidate_execution=a.candidate_execution,generic_tile=dict(block_m=a.generic_bm,block_n=a.generic_bn,pipeline_stages=a.generic_stages),decode_max_tokens=a.decode_max_tokens,flat_history_bindings=a.flat_history_bindings,kernel_source_SHA256=sha(Path(__file__).with_name('two_source_ring.py')),baseline_source_SHA256=sha(a.baseline),results=results,**extra)
 (a.output/'progress.json').write_text(json.dumps(v,indent=2)+'\n')


def fixture(lengths,history_lengths,pad=0,owners=None):
 owners=owners if owners is not None else list(range(len(lengths)))
 columns=22
 total=sum(lengths)+pad
 q=torch.randn(1,HQ,total,D,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(1,HK,total,D,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 smallk=torch.zeros(W,columns,HK,D,device='cuda',dtype=torch.bfloat16);smallv=torch.zeros_like(smallk)
 oldr=W+P-1
 bigk=torch.zeros(oldr,columns,HK,D,device='cuda',dtype=torch.bfloat16);bigv=torch.zeros_like(bigk)
 info=torch.zeros(42,7,device='cuda',dtype=torch.int32)
 rows=[];history={};initial_h={};next_h={};start=0
 for col,n,h in zip(owners,lengths,history_lengths):
  assert n<=P and h>=0
  if col not in history:
   kh=torch.randn(h,HK,D,device='cuda',dtype=torch.bfloat16);vh=torch.randn_like(kh)
   history[col]=(kh,vh);initial_h[col]=h;next_h[col]=h
   if h:
    tail=torch.arange(max(h-W,0),h,device='cuda');smallk[tail%W,col]=kh[tail];smallv[tail%W,col]=vh[tail]
    tail=torch.arange(max(h-oldr,0),h,device='cuda');bigk[tail%oldr,col]=kh[tail];bigv[tail%oldr,col]=vh[tail]
  else:assert h==initial_h[col]
  next_h[col]+=n
  rows.append([start,n,col,next_h[col],columns,0,0]);start+=n
 info[:len(rows)]=torch.tensor(rows,device='cuda',dtype=torch.int32)
 stage=torch.full((3,2,P,HK,D),71.,device='cuda',dtype=torch.bfloat16)
 return dict(stage=stage,initial={'sk':smallk.clone(),'sv':smallv.clone(),'bk':bigk.clone(),'bv':bigv.clone()},q=q,k=k,v=v,sk=smallk,sv=smallv,bk=bigk,bv=bigv,info=info,rows=rows,histories=history,initial_history_lengths=initial_h,lengths=lengths,history_lengths=history_lengths,pad=pad,owners=owners)


def oracle(f):
 out=torch.zeros_like(f['q'])
 for start,n,col,end,_,_,_ in f['rows']:
  kh,vh=f['histories'][col];h=f['initial_history_lengths'][col]
  keep=min(h,W-1);first=h-keep
  indices=[]
  for st,nn,cc,*_ in f['rows']:
   if cc==col:indices.extend(range(st,st+nn))
  ka=torch.cat((kh[-keep:] if keep else kh[:0],f['k'][0,:,indices].transpose(0,1)),0)
  va=torch.cat((vh[-keep:] if keep else vh[:0],f['v'][0,:,indices].transpose(0,1)),0)
  keys=ka.transpose(0,1).repeat_interleave(HQ//HK,0).float()
  vals=va.transpose(0,1).repeat_interleave(HQ//HK,0).float()
  for j in range(0,n,16):
   count=min(16,n-j);query=f['q'][0,:,start+j:start+j+count].float()
   score=torch.matmul(query,keys.transpose(1,2))*SCALE
   kp=torch.arange(first,h+len(indices),device='cuda');qp=torch.arange(end-n+j,end-n+j+count,device='cuda')
   visible=(kp[None,:]<=qp[:,None])&(kp[None,:]>qp[:,None]-W)
   probs=score.masked_fill(~visible[None],-float('inf')).softmax(-1)
   out[0,:,start+j:start+j+count]=torch.matmul(probs,vals).to(torch.bfloat16)
 return out


def old_write(f):
 # Same scatter kernel, with large original ring capacity, commits all rows.
 ring_commit_current(f['bk'],f['bv'],f['k'],f['v'],f['info'],W+P-1)


def baseline_tensor_step(q,bk,bv,k,v,info):
 ring_commit_current(bk,bv,k,v,info,W+P-1)
 return mod.sdpa_varlen(q,bk,bv,info,1*SCALE,True,W,W+P-1)
baseline_step=torch.compile(baseline_tensor_step,fullgraph=True,dynamic=False) if a.baseline_execution=='compiled' else baseline_tensor_step
def baseline(f):
 return baseline_step(f['q'],f['bk'],f['bv'],f['k'],f['v'],f['info'])


def compiled_tensor_step(q,sk,sv,k,v,info,stage):
 kh=sk.view(W,22,HK,D) if a.flat_history_bindings else sk
 vh=sv.view(W,22,HK,D) if a.flat_history_bindings else sv
 out=sdpa_ring_current(q,kh,vh,k,v,info,W,1*SCALE,a.generic_bm,a.generic_bn,a.generic_stages,a.decode_max_tokens)
 ring_commit_journal(sk,sv,k,v,info,W,stage,1)
 return out
compiled_step=torch.compile(compiled_tensor_step,fullgraph=True,dynamic=False) if a.candidate_execution=='compiled' else compiled_tensor_step

def candidate(f,commit=True):
 assert commit
 sk=f['sk'].view(-1) if a.flat_history_bindings else f['sk']
 sv=f['sv'].view(-1) if a.flat_history_bindings else f['sv']
 return compiled_step(f['q'],sk,sv,f['k'],f['v'],f['info'],f['stage'])


def errors(x,y,total):
 d=(x[:,:,:total].float()-y[:,:,:total].float()).flatten()
 ref=y[:,:,:total].float().flatten()
 return dict(max_abs=float(d.abs().max()),normalized_RMS=float(d.square().mean().sqrt()/(ref.square().mean().sqrt()+1e-12)))


def check_commit(f):
 final={}
 for row in f['rows']:final[row[2]]=max(final.get(row[2],0),row[3])
 for start,n,col,end,_,_,_ in f['rows']:
  first=max(end-n,final[col]-W)
  if first>=end:continue
  ts=torch.arange(start+first-(end-n),start+n,device='cuda');pos=torch.arange(first,end,device='cuda')%W
  assert torch.equal(f['sk'][pos,col],f['k'][0,:,ts].transpose(0,1))
  assert torch.equal(f['sv'][pos,col],f['v'][0,:,ts].transpose(0,1))
 assert torch.count_nonzero(f['sk'][:,-1]).item()==0
 assert torch.count_nonzero(f['sv'][:,-1]).item()==0
 real=sum(f['lengths'])
 assert torch.equal(f['stage'][1,0,:real],f['k'][0,:,:real].transpose(0,1))
 assert torch.equal(f['stage'][1,1,:real],f['v'][0,:,:real].transpose(0,1))
 assert torch.all(f['stage'][0]==71) and torch.all(f['stage'][2]==71)
 assert torch.all(f['stage'][1,:,real:]==71)


def dynamic_graph_check(f):
 for key in ['sk','sv']:f[key].copy_(f['initial'][key])
 for _ in range(2):candidate(f)
 torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=candidate(f)
 g=fixture(f['lengths'],[x+17 for x in f['history_lengths']],f['pad'],f['owners'])
 reference=oracle(g)
 for key in ['q','k','v','info']:f[key].copy_(g[key])
 for key in ['sk','sv']:f[key].copy_(g['initial'][key])
 graph.replay();torch.cuda.synchronize()
 e=errors(out,reference,sum(g['lengths']));assert e['normalized_RMS']<.01,e
 g['sk']=f['sk'];g['sv']=f['sv'];g['stage']=f['stage'];check_commit(g)
 return dict(changed_metadata_and_inputs_without_recapture=True,error=e,commit_exact=True,journal_exact=True)


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

cases=[('empty-d1',[1],[0],0),('history-d1',[1],[W*3+7],0),('d8',[1]*8,[W*2+i*13 for i in range(8)],0),('d16',[1]*16,[W*3+i*31 for i in range(16)],0),('p64',[64],[W*2+37],0),('p512',[512],[W*3+11],0),('max-prefill',[P],[W*2+17],0),('mixed64',[1]*8+[64],[W*3+i*11 for i in range(9)],0),('mixed512',[1]*8+[512],[W*2+i*7 for i in range(9)],0),('unequal-tail-pad',[1,3,17],[0,W-1,W+7],43),('repeated-owner-small',[3,2,17],[W*2+11,W+23,W*2+11],0),('repeated-owner-wrap',[W//2,1,W//2+17],[W*2+11,W+23,W*2+11],0)]
try:
 record('running-kernel-prerequisite')
 for name,lengths,hist,pad in cases:
  if name not in ['empty-d1','history-d1','d16','p64','max-prefill','unequal-tail-pad'] and not (a.include_d8 and name=='d8'):continue
  assert sum(lengths)<=P
  owners=[0,1,0] if name.startswith('repeated-owner') else None
  f=fixture(lengths,hist,pad,owners);ref=oracle(f)
  old=baseline(f);new=candidate(f);torch.cuda.synchronize()
  oe=errors(old,ref,sum(lengths));ne=errors(new,ref,sum(lengths));delta=errors(new,old,sum(lengths))
  assert ne['normalized_RMS']<=max(.01,3*oe['normalized_RMS']),dict(case=name,candidate=ne,baseline=oe)
  check_commit(f)
  t=timing(f)
  graph_check=dynamic_graph_check(f)
  results.append(dict(case=name,lengths=lengths,history_lengths=hist,padding=pad,graph_replay_check=graph_check,baseline_vs_FP32_oracle=oe,candidate_vs_FP32_oracle=ne,candidate_vs_baseline=delta,commit_exact=True,journal_exact=True,other_layers_and_stage_padding_untouched=True,padding_writes_isolated=True,attention_and_commit_timing=t,history_bytes_baseline=f['bk'].numel()*f['bk'].element_size()*2,history_bytes_candidate=f['sk'].numel()*f['sk'].element_size()*2))
  record('running-kernel-prerequisite',last_completed=name)
  print(json.dumps({'case':name,'candidate_RMS':ne['normalized_RMS'],'baseline_ms':t['baseline']['retained_mean_ms'],'candidate_ms':t['candidate']['retained_mean_ms']}),flush=True)
  del f,ref,old,new
  torch.cuda.empty_cache()
 record('kernel-prerequisite-collected',kernel_numerical_gate_passed=True,compiled_journal_gate=a.candidate_execution=='compiled',ET_mini_model_qualified=False,allocator_policy_qualified=False,graph_dynamic_metadata_qualified=True,native_ring_storage_qualified=False,scope='Actual CUDA kernel prerequisites only; four-condition mini model/native qualification remains required.')
 (a.output/'result.json').write_text((a.output/'progress.json').read_text())
except BaseException as e:
 record('failed-kernel-prerequisite',error=repr(e),traceback=traceback.format_exc());raise
