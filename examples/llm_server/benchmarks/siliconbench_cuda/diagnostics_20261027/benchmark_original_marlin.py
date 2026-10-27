"""Original installed Marlin versus unchanged ET kernels on the same GGUF codes.

Repack once, never re-quantize weight codes. Integer and floating zero-point
representations are audited separately. FP16 is an explicit A16 experiment,
not a claim of bit-identical BF16 arithmetic or ET server integration.
"""
import argparse,ast,datetime,fcntl,hashlib,json,os,statistics,subprocess,sys,time,traceback,types
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_functions(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign,ast.AnnAssign)) and (isinstance(n,ast.FunctionDef) and n.name in names)]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['et-source','gguf','output','lock']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--model',choices=['MG','Gemma'],required=True);p.add_argument('--max-seconds',type=int,default=7200)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 report=dict(scope=__doc__,model=a.model,cases=[],packing=[],unsupported_GGUF_tensors=[],new_KA=False,new_static_methods=False,ET_source_modified=False,ET_server_integration_qualified=False)
 def state(s,**kw):
  (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  t=a.output/'progress.pending.json';t.write_text(json.dumps(dict(status=s,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2)+'\n');t.replace(a.output/'progress.json')
 try:
  state('waiting-exclusive-GPU')
  with a.lock.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   start=time.monotonic()
   os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',TRITON_CACHE_DIR=str(a.output/'triton-cache'),TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor-cache'))
   import numpy as np,torch,triton
   from gguf import GGUFReader,GGMLQuantizationType
   import vllm
   from vllm import _custom_ops as ops
   from vllm.scalar_type import scalar_types
   from vllm.model_executor.layers.quantization.utils import marlin_utils as marlin
   from torch._library.triton import set_wrap_triton_enabled
   # Kernel-only namespace imports skip ET export/compiler registration. Every
   # actual kernel/helper source remains unchanged, and is hashed below.
   for name,rel in [('executorch',''),('executorch.backends','backends'),('executorch.backends.cuda','backends/cuda'),('executorch.backends.cuda.triton','backends/cuda/triton'),('executorch.backends.cuda.triton.kernels','backends/cuda/triton/kernels'),('executorch.backends.cuda.autotune','backends/cuda/autotune')]:
    m=types.ModuleType(name);m.__path__=[str(a.et_source/rel)];sys.modules[name]=m
   from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
   from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
   here=Path(__file__).resolve().parent.parent
   sys.path.insert(0,str(here))
   import gemma_runtime_precision as gemma
   assert sha(native.__file__)=='fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
   assert sha(gemma.__file__)=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
   torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
   assert torch.cuda.get_device_capability()==(12,0)
   expected='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e' if a.model=='MG' else '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
   assert sha(a.gguf)==expected
   report.update(GGUF_SHA256=expected,torch=torch.__version__,vllm=vllm.__version__,GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,power.limit,driver_version','--format=csv,noheader'],text=True).strip(),native_kernel_SHA256=sha(native.__file__),Gemma_kernel_SHA256=sha(gemma.__file__),installed_Marlin_utils_SHA256=sha(marlin.__file__),timing_policy='CUDA graph GPU events; 4 alternating rounds, discard round 1; numerical checks precede timings; repack excluded but incremental memory reported')
   import typing
   env=dict(torch=torch,Tensor=torch.Tensor,Tuple=typing.Tuple,Optional=typing.Optional,QK_K=256,Q4_K_GROUP_SIZE=32,_Q4_K_BLOCK_BYTES=144,_SUPER_BLOCK=256,_CODE_MAX=255)
   load_functions(a.et_source/'backends/cuda/coalesced_int4_tensor.py',['_encode_uint8_per_super'],env)
   load_functions(a.et_source/'extension/llm/export/gguf.py',['_read_f16','_q4_k_fields'],env)
   def portable(q,s,z,gs,dtype):return types.SimpleNamespace(qdata=q,scale=s,zero_point=z,group_size=gs,orig_dtype=dtype)
   env['ExportableInt4Tensor']=portable
   load_functions(here/'gemma_runtime_python/batching_gguf_loader.py',['decode_q4_blocks'],env)
   report['packing_source_SHA256']={str(x.relative_to(a.et_source)):sha(x) for x in [a.et_source/'backends/cuda/coalesced_int4_tensor.py',a.et_source/'extension/llm/export/gguf.py']}
   report['packing_source_SHA256']['Gemma_decode']=sha(here/'gemma_runtime_python/batching_gguf_loader.py')
   reader=GGUFReader(str(a.gguf))
   def unpack(role):
    t=next(t for t in reader.tensors if t.name=='blk.0.'+role+'.weight');n,k=map(int,t.shape[::-1]);raw=torch.from_numpy(np.array(t.data,copy=True)).view(torch.uint8).reshape(n,-1)
    if a.model=='Gemma':
     assert t.tensor_type==GGMLQuantizationType.Q4_0
     out=env['decode_q4_blocks'](raw,'q4_0',(n,k))
    else:
     assert t.tensor_type==GGMLQuantizationType.Q4_K
     q,s,minv=env['_q4_k_fields'](raw,n,k);z=torch.where(s!=0,minv/s,torch.zeros_like(minv))
     out=portable(q[:,::2]|(q[:,1::2]<<4),s.bfloat16().t().contiguous(),z.bfloat16().t().contiguous(),32,torch.bfloat16)
    return out,dict(name=t.name,N=n,K=k,raw_SHA256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())
   def stats(y,ref):
    d=y.float()-ref.float();return dict(relative_rms=float(d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-8)),max_abs=float(d.abs().max()),finite=bool(torch.isfinite(y).all()),different_elements=int((y!=ref).sum()))
   def bounded():assert time.monotonic()-start<a.max_seconds,'Bounded experiment time exceeded'
   def graph_time(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph();begin=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
    with torch.cuda.graph(g):
     begin.record();y=fn();end.record()
    out=[]
    for _ in range(15):g.replay();torch.cuda.synchronize();out.append(begin.elapsed_time(end)*1000)
    return statistics.median(out)
   with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()),set_wrap_triton_enabled(False):
    for role in ['ffn_gate_up','ffn_down']:
     bounded();state('packing-exact-GGUF-codes',role=role)
     names=['ffn_gate','ffn_up'] if role=='ffn_gate_up' else [role]
     raw_tensors=[next(t for t in reader.tensors if t.name=='blk.0.'+name+'.weight') for name in names]
     supported_type=GGMLQuantizationType.Q4_0 if a.model=='Gemma' else GGMLQuantizationType.Q4_K
     unsupported=[dict(name=t.name,type=t.tensor_type.name,shape=list(map(int,t.shape)),action='Keep existing ET path; do not requantize') for t in raw_tensors if t.tensor_type!=supported_type]
     if unsupported:
      report['unsupported_GGUF_tensors'].extend(unsupported);state('recorded-mixed-GGUF-uncovered-route',role=role);continue
     if role=='ffn_gate_up':
      values=[unpack(n) for n in ['ffn_gate','ffn_up']];ps=[v[0] for v in values]
      obj=portable(torch.cat([p.qdata for p in ps],0),torch.cat([p.scale for p in ps],1),torch.cat([p.zero_point for p in ps],1),32,torch.bfloat16);source=[v[1] for v in values]
     else:obj,src=unpack(role);source=[src]
     sc,ss=env['_encode_uint8_per_super'](obj.scale,32);zc,zs=env['_encode_uint8_per_super'](obj.zero_point,32)
     w=tuple(t.cuda().contiguous() for t in [obj.qdata,sc,ss,zc,zs]);q,sc,ss,zc,zs=w;n,k=q.shape[0],q.shape[1]*2
     codes=torch.stack((q&15,q>>4),-1).reshape(n,k).to(torch.int64)
     scales=sc.float()*ss.float().repeat_interleave(8,1);zeros=zc.float()*zs.float().repeat_interleave(8,1)
     fp32dense=((codes.float().reshape(n,k//32,32)-zeros[:,:,None])*scales[:,:,None]).reshape(n,k).bfloat16()
     # Exact original GPTQ packing: K/8 rows, N columns; no change to codes.
     qt=codes.t().reshape(k//8,8,n)
     packed=torch.sum(qt*(1<<(4*torch.arange(8,device='cuda',dtype=torch.int64)))[None,:,None],1).to(torch.int32).contiguous()
     marlinq=ops.gptq_marlin_repack(packed,k,n,4)
     workspace=marlin.marlin_make_workspace_new(torch.device('cuda',0),marlin.MARLIN_MAX_BLOCKS_PER_SM)
     packing=dict(role=role,sources=source,N=n,K=k,ET_packed_SHA256=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in w],codes_unchanged=True,repack_once=True,Marlin_weight_bytes=marlinq.numel()*marlinq.element_size(),fractional_zero_groups=int((zeros!=zeros.round()).sum()),max_zero_rounding=float((zeros-zeros.round()).abs().max()),floating_zp_requires_FP16_in_reference=True)
     report['packing'].append(packing);state('running-numerics-before-timing',role=role)
     def dqlinear(x,q,sc,ss,z,zs):
      nn,kk=q.shape[0],q.shape[1]*2
      c=torch.stack(((q&15).float(),(q>>4).float()),-1).reshape(nn,kk//32,32)
      s=sc.float()*ss.float().repeat_interleave(8,1);zz=z.float()*zs.float().repeat_interleave(8,1)
      dense=((c-zz[:,:,None])*s[:,:,None]).reshape(nn,kk).bfloat16()
      return torch.nn.functional.linear(x,dense)
     fallback=torch.compile(dqlinear,fullgraph=True,dynamic=False,options=dict(max_autotune_gemm=True,max_autotune_gemm_backends='TRITON',emulate_precision_casts=True,**{'triton.cudagraphs':False}))
     base=gemma if a.model=='Gemma' else native
     methods={}
     # Integer zero points are approximate for the current secondary metadata;
     # never label these bit-exact merely because the GGUF codes are preserved.
     for name,dtype,floating in [('Marlin-BF16-integer-zp',torch.bfloat16,False),('Marlin-FP16-float-zp',torch.float16,True)]:
      scale=marlin.marlin_permute_scales(scales.t().to(dtype).contiguous(),k,n,32)
      zero=marlin.marlin_permute_scales(zeros.t().to(dtype).contiguous(),k,n,32) if floating else marlin.marlin_zero_points(zeros.round().t().to(torch.int32).contiguous(),k//32,n,4)
      methods[name]=(dtype,scale,zero,floating)
     # Preserve the fractional zero offset using an explicit group-sum
     # correction. Marlin still rounds scales to BF16; this is a numerical
     # candidate, never an assertion of bit-identical ET arithmetic.
     # x @ ((q-z)*s) = Marlin(x,q,round(z),s)
     #                  + sum_group(x) @ ((round(z)-z)*s).
     zero_residual=((zeros.round()-zeros)*scales.to(torch.bfloat16).float()).t().contiguous()
     def compensate(x,y,residual):
      grouped=x.float().reshape(x.shape[0],x.shape[1]//32,32).sum(dim=2)
      return (y.float()+torch.mm(grouped,residual)).to(x.dtype)
     compensation=torch.compile(compensate,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
     packing['zero_correction_bytes']=zero_residual.numel()*zero_residual.element_size()
     packing['BF16_scale_rounding_retained']=True
     sizes=[8,16,64,128,512,2048]+([4096] if a.model=='MG' else [])
     for m in sizes:
      bounded();state('validating-and-timing',role=role,M=m)
      x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(27000+m+k))
      oracle=torch.nn.functional.linear(x.float(),fp32dense.float()).bfloat16()
      calls={'dense-BF16-linear-control':lambda x=x:torch.nn.functional.linear(x,fp32dense)}
      record=dict(role=role,M=m,N=n,K=k,checks={},timings={},ET_control_kernel_route=('W4A16' if m<=64 else 'DQ-BF16-linear'),activation_BF16_source=True,ET_source_methods_unchanged=True)
      if a.model=='Gemma' and m in [128,512]:record['control_limitation']='Generic fallback only; does not measure selected prepared Gemma prefill port at this M'
      if m<=64:
       with autotune_launch_params() as selection:base._launch_w4a16(m,x,*w,32)
       assert len(selection.picks)==1
       split=int(selection.picks[0].choice);kernels=base._W4A16_DIRECT_KERNELS if split==1 else base._W4A16_SPLIT_KERNELS;cfg=kernels[m].best_config
       calls['ET-unchanged-static-W4A16']=lambda x=x,m=m,split=split,cfg=cfg:base._launch_w4a16(m,x,*w,32,split_k=split,config=cfg)
       record['ET_tuning']=dict(split_k=split,split_times_us=selection.picks[0].times_us,config=dict(kwargs=cfg.kwargs,num_warps=cfg.num_warps,num_stages=cfg.num_stages))
      else:calls['ET-compiled-DQ-BF16-linear']=lambda x=x:fallback(x,*w)
      for name,(dtype,s,z,floating) in methods.items():
       xx=x.to(dtype)
       call=lambda xx=xx,s=s,z=z,floating=floating:marlin._marlin_gemm(xx,marlinq,None,s,None,None,z,workspace,scalar_types.uint4.id,m,n,k,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=floating)
       try:
        y=call();again=call();torch.cuda.synchronize();check=stats(y,oracle);check.update(deterministic=bool(torch.equal(y,again)),BF16_to_FP16_input_cast=dtype==torch.float16,integer_zero_rounded=not floating)
        # Numerical gate is a screening criterion; a full model task gate is
        # still required before applying a representation change to ET.
        check['numeric_screen_pass']=check['finite'] and check['relative_rms']<.005 and check['deterministic']
        record['checks'][name]=check
        if check['numeric_screen_pass']:calls[name]=call
        if not floating:
         corrected=lambda call=call,x=x:compensation(x,call(),zero_residual)
         cy=corrected();ca=corrected();torch.cuda.synchronize();cs=stats(cy,oracle)
         cs.update(deterministic=bool(torch.equal(cy,ca)),integer_zero_rounding_compensated=True,scale_BF16_rounding_retained=True,extra_FP32_group_sum_and_matmul=True)
         cs['numeric_screen_pass']=cs['finite'] and cs['relative_rms']<.005 and cs['deterministic']
         record['checks']['Marlin-BF16-zero-compensated']=cs
         if cs['numeric_screen_pass']:calls['Marlin-BF16-zero-compensated']=corrected
       except Exception as e:
        failed_name='Marlin-BF16-zero-compensated' if name in record['checks'] else name
        record['checks'][failed_name]=dict(supported=False,error_type=type(e).__name__,detail=str(e)[-1000:])
      for name,fn in calls.items():
       y=fn();torch.cuda.synchronize();record['checks'].setdefault(name,stats(y,oracle))
      report['cases'].append(record);state('timing-screened-candidates',role=role,M=m)
      samples={name:[] for name in calls}
      for round in range(4):
       names=list(calls)
       if round%2:names.reverse()
       for name in names:bounded();samples[name].append(graph_time(calls[name]))
      record['timings']={name:dict(samples_us=v,mean_last3_us=statistics.mean(v[1:]),SD_last3_us=statistics.stdev(v[1:])) for name,v in samples.items()}
      state('case-collected',role=role,M=m)
     del w,q,sc,ss,zc,zs,codes,scales,zeros,fp32dense,packed,marlinq,workspace,calls,methods,obj,zero_residual,compensation;torch.cuda.empty_cache()
   state('collected-pending-equivalence-and-ET-integration-review',cases=len(report['cases']),ET_server_integration_qualified=False)
 except Exception as e:
  (a.output/'exception.txt').write_text(traceback.format_exc());state('failed-preserve-evidence',error_type=type(e).__name__,detail=str(e)[-1500:]);raise
if __name__=='__main__':main()
