"""Export same-GGUF cropped MG/Gemma baseline and split-ring mini artifacts."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,re,sys,traceback
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(16*2**20):h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for n in ['native-preparation','gguf','output','lock']:a.add_argument('--'+n,type=Path,required=True)
 a.add_argument('--model',choices=['MG','Gemma'],required=True);a.add_argument('--reuse-baseline-audit',type=Path);a.add_argument('--only-arm',choices=['baseline','ring']);a.add_argument('--cache-root',type=Path);args=a.parse_args()
 assert not args.output.exists();args.output.mkdir(parents=True)
 gate=json.loads((args.native_preparation/'native-gate.json').read_text())
 assert gate['native_tests_passed'] and gate['legacy_test_count']>=74 and gate['allocator_policy_test_count']==4
 assert not gate['real_mini_model_qualified']
 source=Path(gate['export_source']);audit=json.loads((source/'ring-retention-source-audit.json').read_text())
 for name,h in {**audit['candidate_SHA256'],**audit.get('additional_candidate_SHA256',{})}.items():assert digest(source/name)==h,name
 expected={'MG':'4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e','Gemma':'7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'}[args.model]
 assert digest(args.gguf)==expected
 package=Path(__file__).parent;parent=package.parent;budget=4096 if args.model=='MG' else 2048
 def state(status,**extra):
  p=args.output/'progress.pending.json';p.write_text(json.dumps(dict(status=status,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=args.model,real_mini_qualified=False,full_model_export_started=False,**extra),indent=2)+'\n');p.replace(args.output/'progress.json')
 try:
  state('waiting-mini-export-GPU-lock')
  with args.lock.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   import subprocess
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='12.0',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1')
   sys.path.insert(0,str(parent));sys.path.insert(0,str(source.parent))
   import torch
   import torch._inductor.cpp_builder as cpp
   from dataclasses import replace
   from executorch.backends.cuda.cuda_backend import CudaBackend
   assert torch.cuda.get_device_capability()==(12,0);torch.set_num_threads(2)
   orig=CudaBackend.get_aoti_compile_options
   def opts(cls,specs):
    o=dict(orig(specs));o['aot_inductor.compile_wrapper_opt_level']='O2';return o
   CudaBackend.get_aoti_compile_options=classmethod(opts)
   origbuild=cpp.CppBuilder.build;current=None
   def build(builder):
    flags=re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)',builder.get_command_line())
    with (current/'wrapper-flags.jsonl').open('a') as f:f.write(json.dumps(dict(flags=flags,target=builder._name))+'\n')
    return origbuild(builder)
   cpp.CppBuilder.build=build
   if args.model=='MG':
    from executorch.examples.models.muse_glimmer.export import export_solo as exporter
    from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader as loader
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig
    full=MuseGlimmerConfig(max_seq_len=131072);config=replace(full,n_layers=4)
    def make_model():
     atomic=loader._atomic_sd_from_gguf(str(args.gguf),torch.bfloat16);kept={}
     for name,value in atomic.items():
      m=re.match(r'^layers\.(\d+)\.',name)
      if m is None or int(m.group(1))<4:kept[name]=value
     del atomic
     return loader._finalize(kept,'cuda',config,torch.bfloat16)
    def export_model(model,path):
     exporter.export_cuda_batching(model,config,str(path),max_step_tokens=budget,max_cells=65536,forward_sizes=[1,2,4,8,16,32,64],device_sampling=True,kv_layout='sequence',max_kv_seqs=21,runtime_kv_columns=True,independent_capacity=True)
   else:
    from executorch.examples.models.gemma4_31b.model import Gemma4_31B,Gemma4_31BConfig,materialize_runtime_buffers
    from executorch.examples.models.gemma4_31b.batching_model import PackedGemma4
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import iter_gemma_gguf
    from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key
    from executorch.examples.models.gemma4_31b.cuda_packers import convert_quantized_tensors_for_cuda
    from executorch.examples.models.gemma4_31b.cuda_source_transformations import cuda_source_transformations
    from executorch.examples.models.gemma4_31b.export_batching import export_cuda_batching
    from executorch.extension.llm.export.load import load_checkpoint
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    import executorch.backends.cuda.quantize_op_dispatch.int4_dispatch as dispatch
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import chunked_dequant_linear
    from public_prefill_gemma_export_hook import install_gemma_prefill_port
    precision_path=parent/'gemma_runtime_precision.py';assert digest(precision_path)=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
    # The exact pinned precision module is shared with prior accepted exports.
    spec=importlib.util.spec_from_file_location('Gemma_precision',precision_path);precision=importlib.util.module_from_spec(spec);spec.loader.exec_module(precision)
    native.INT4_QUANTIZED_GEMM=precision.INT4_QUANTIZED_GEMM
    def dequant_linear(x,qdata,scale,scale_step,zero,zero_step,group_size):
     n,khalf=qdata.shape;k=khalf*2;groups,gps=k//group_size,256//group_size
     def rows(i,j):
      packed=qdata[i:j].reshape(j-i,groups,group_size//2)
      q=torch.stack(((packed&15).float(),(packed>>4).float()),dim=-1).reshape(j-i,groups,group_size)
      s=scale[i:j].float()*scale_step[i:j].float().repeat_interleave(gps,dim=1)
      z=zero[i:j].float()*zero_step[i:j].float().repeat_interleave(gps,dim=1)
      return torch.nn.functional.linear(x,((q-z[:,:,None])*s[:,:,None]).reshape(j-i,k).to(x.dtype))
     return chunked_dequant_linear(x,n,rows)
    dispatch._dequant_matmul=dequant_linear
    assert install_gemma_prefill_port(dispatch,capability=(12,0),enabled=True,runtime_chunk_range=True)['installed']
    full=Gemma4_31BConfig(hidden_size=3840,intermediate_size=15360,num_hidden_layers=48,num_attention_heads=16,num_key_value_heads=8,num_global_key_value_heads=1,max_seq_len=131072)
    config=replace(full,num_hidden_layers=6,layer_types=full.layer_types[:6]);assert config.layer_types==['sliding_attention']*5+['full_attention']
    def retained_iter(path):
     for name,t in iter_gemma_gguf(path):
      key=gguf_to_model_key(name);m=re.match(r'^layers\.(\d+)\.',key or '')
      if m is None or int(m.group(1))<6:yield name,t
    def make_model():
     with torch.device('meta'):model=Gemma4_31B(config)
     load_checkpoint(str(args.gguf),model,key_map=gguf_to_model_key,tie_map={'embed_tokens.weight':('lm_head.weight',True)},dtype=torch.bfloat16,raw_iter=retained_iter)
     materialize_runtime_buffers(model,dtype=torch.bfloat16);convert_quantized_tensors_for_cuda(model);cuda_source_transformations(model,use_turboquant=False)
     return PackedGemma4(model).eval()
    def export_model(model,path):
     export_cuda_batching(model,str(path),max_step_tokens=budget,max_kv_seqs=21,forward_sizes=(1,2,4,8,16,32,64),suppressed_token_ids=(258882,258883),runtime_kv_columns=True,independent_capacity=True)
   from executorch.exir._serialize._program import deserialize_pte_binary
   from executorch.exir.schema import Int,String
   records=[]
   for arm in ([args.only_arm] if args.only_arm else ['baseline','ring']):
    if arm=='baseline' and args.reuse_baseline_audit:
     row=json.loads(args.reuse_baseline_audit.read_text());assert row['arm']=='baseline' and row['model']==args.model and row['prefill_budget']==budget and row['GGUF_SHA256']==expected
     for n,v in row['artifacts'].items():assert digest(Path(row['artifact'])/n)==v['SHA256']
     row=dict(row,reused_immutable_baseline_audit=str(args.reuse_baseline_audit),reused_immutable_baseline_audit_SHA256=digest(args.reuse_baseline_audit))
     (args.output/'baseline').mkdir();(args.output/'baseline/artifact-audit.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row);continue
    current=args.output/arm;current.mkdir();cache=current/'cache';(cache/'host-zero').mkdir(parents=True)
    compile_cache=(args.cache_root/arm) if args.cache_root else cache
    os.environ.update(ET_CUDA_SPLIT_RING_HISTORY='1' if arm=='ring' else '0',TORCHINDUCTOR_CACHE_DIR=str(compile_cache/'inductor'),TRITON_CACHE_DIR=str(compile_cache/'triton'),ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache/'host-zero'))
    state('exporting-real-mini-'+arm,source=str(source),prefill_budget=budget)
    model=make_model();artifact=current/'artifact';export_model(model,artifact);del model
    program=deserialize_pte_binary((artifact/'model.pte').read_bytes()).program;methods={p.name:p for p in program.execution_plan};metadata={}
    for n,p in methods.items():
     if n.startswith('get_') and len(p.outputs)==1:
      v=p.values[p.outputs[0]].val
      if isinstance(v,Int):metadata[n]=v.int_val
      elif isinstance(v,String):metadata[n]=v.string_val
    assert metadata['get_offgraph_kv_sequence_info_abi']==(4 if arm=='ring' else 3)
    assert metadata['get_max_prefill_chunk']==budget and metadata['get_max_context_len']==131072
    assert metadata['get_n_caches']==(4 if args.model=='MG' else 6)
    manifests={}
    for n,p in methods.items():
     if not n.startswith('forward_'):continue
     for d in p.delegates:
      for spec in d.compile_specs:
       if spec.key=='offgraph_kv_manifest':
        m=json.loads(bytes(spec.value));assert m['version']==(4 if arm=='ring' else 3);assert m.get('split_ring_history',False)==(arm=='ring');assert m['max_write']==budget;manifests[n]=m
    assert len(manifests)==8
    row=dict(arm=arm,model=args.model,retained_original_layers=list(range(metadata['get_n_caches'])),GGUF_SHA256=expected,quantization_unchanged=True,BF16_KV=True,bulk_prefill_A16=True,prefill_budget=budget,metadata=metadata,manifests=manifests,artifact=str(artifact),artifacts={n:dict(SHA256=digest(artifact/n),bytes=(artifact/n).stat().st_size) for n in ['model.pte','aoti_cuda_blob.ptd']},real_mini_qualified=False)
    (current/'artifact-audit.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row)
    torch.cuda.synchronize();torch.cuda.empty_cache()
   result=dict(model=args.model,arms=records,native_gate_SHA256=digest(args.native_preparation/'native-gate.json'),real_mini_qualified=False,model_numerics_memory_latency_gates_pending=True)
   (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');state('real-mini-artifacts-exported',arms=records,only_arm=args.only_arm)
 except BaseException as e:state('failed-real-mini-export',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
