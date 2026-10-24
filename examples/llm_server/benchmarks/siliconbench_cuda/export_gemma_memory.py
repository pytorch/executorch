"""Export a separate Gemma artifact with a smaller, safe prefill write budget.

Keep GGUF weights, context, session capacity, static forward buckets and the
validated precision implementation. This is a candidate memory/performance
tradeoff, not an accepted runtime or automatic benchmark replacement.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        while chunk:=file.read(16*1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--et-source',type=Path,required=True)
    parser.add_argument('--gguf',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--precision-control',type=Path,required=True)
    parser.add_argument('--max-step-tokens',type=int,choices=[256,512,1024,2048],default=512)
    parser.add_argument('--max-kv-seqs',type=int,choices=[17,18,33],default=18)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--owned-gpu-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-context', type=int, choices=[12288,131072], default=12288)
    args=parser.parse_args()

    def state(status,**fields):
        pending=args.job_dir/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
                                         max_step_tokens=args.max_step_tokens,
                                         max_kv_seqs=args.max_kv_seqs,**fields),indent=2)+'\n')
        pending.replace(args.job_dir/'progress.json')

    if not args.owned_gpu_worker:
        assert not args.job_dir.exists() and not args.output.exists(),'Preserve existing exports and attempts'
        args.job_dir.mkdir(parents=True)
        (args.job_dir/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        state('waiting-GPU-lock')
        try:
            with args.lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                with (args.job_dir/'export.log').open('x') as log:
                    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                                    *sys.argv[1:],'--owned-gpu-worker'],stdout=log,
                                   stderr=subprocess.STDOUT,check=True,timeout=21600)
                assert json.loads((args.job_dir/'progress.json').read_text())['status']=='artifact-exported-and-audited'
                deadline=time.monotonic()+60
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    assert time.monotonic()<deadline,'Wait for completed export CUDA context teardown'
                    time.sleep(1)
                state('completed',artifact=str(args.output),
                      performance_and_GMRID_acceptance_pending=True,no_runtime_replacement=True)
        except Exception as exc:
            state('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:])
            raise
        return

    assert json.loads((args.job_dir/'lock-owner.json').read_text())['pid']==os.getppid()
    with args.lock.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('GPU worker requires its parent to own the GPU lock')
    assert digest(args.gguf)=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
    assert digest(args.precision_control)=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
    assert not args.output.exists()
    cache=args.job_dir/'cache'
    (cache/'host-zero-backed').mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',
                      TORCH_CUDA_ARCH_LIST='12.0',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                      ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache/'host-zero-backed'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                      OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(args.et_source.parent))
    import torch
    import torch._inductor.cpp_builder as cpp
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    import executorch.backends.cuda.quantize_op_dispatch.int4_dispatch as dispatch
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import chunked_dequant_linear
    from executorch.examples.models.gemma4_31b import cuda_packers
    assert digest(Path(cuda_packers.__file__))=='c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
    assert torch.cuda.get_device_capability()==(12,0)
    spec=importlib.util.spec_from_file_location('verified_export_precision',args.precision_control)
    corrected=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corrected)
    native.INT4_QUANTIZED_GEMM=corrected.INT4_QUANTIZED_GEMM

    def dequant_linear(x,qdata,scale,scale_step,zero,zero_step,group_size):
        n,khalf=qdata.shape;k=khalf*2;groups=k//group_size;gps=256//group_size
        def rows(i,j):
            packed=qdata[i:j].reshape(j-i,groups,group_size//2)
            q=torch.stack(((packed&15).float(),(packed>>4).float()),dim=-1).reshape(j-i,groups,group_size)
            s=scale[i:j].float()*scale_step[i:j].float().repeat_interleave(gps,dim=1)
            z=zero[i:j].float()*zero_step[i:j].float().repeat_interleave(gps,dim=1)
            dense=((q-z[:,:,None])*s[:,:,None]).reshape(j-i,k).to(x.dtype)
            return torch.nn.functional.linear(x,dense)
        return chunked_dequant_linear(x,n,rows)
    dispatch._dequant_matmul=dequant_linear
    original_build=cpp.CppBuilder.build
    def record_build(self):
        flags=re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)',self.get_command_line())
        with (args.job_dir/'actual-compiler-optimization.jsonl').open('a') as out:
            out.write(json.dumps(dict(builder=type(self).__name__,optimization_flags=flags,target_name=self._name))+'\n')
        return original_build(self)
    cpp.CppBuilder.build=record_build
    state('exporting',safe_ring_rows=1024+args.max_step_tokens-1)
    sys.argv=['export_batching','--gguf',str(args.gguf),'--config',str(args.config),
              '--output-dir',str(args.output),'--max-seq-len',str(args.max_context),
              '--max-step-tokens',str(args.max_step_tokens),'--max-kv-seqs',str(args.max_kv_seqs)]
    runpy.run_module('executorch.examples.models.gemma4_31b.export_batching',run_name='__main__')

    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import Int, String
    from executorch.exir.scalar_type import ScalarType
    program=deserialize_pte_binary((args.output/'model.pte').read_bytes()).program
    methods={plan.name:plan for plan in program.execution_plan}
    forwards={f'forward_{m}' for m in [1,2,4,8,16,32,64]}|{'forward_others'}
    assert forwards|{'sample','sample_argmax'} <= set(methods)
    assert not any(name.startswith('draft_') for name in methods) and 'vision_encoder' not in methods
    metadata={}
    for name,plan in methods.items():
        if name.startswith('get_') and len(plan.outputs)==1:
            value=plan.values[plan.outputs[0]].val
            if isinstance(value,Int):metadata[name]=value.int_val
            elif isinstance(value,String):metadata[name]=value.string_val
    assert metadata['get_offgraph_kv_max_seqs']==args.max_kv_seqs and metadata['get_max_context_len']==args.max_context
    assert metadata['get_activation_dtype'] in [int(ScalarType.BFLOAT16),'bf16','bfloat16']
    assert metadata['get_n_caches']==48 and metadata['get_max_prefill_chunk']==args.max_step_tokens
    output=methods['forward_others'].values[methods['forward_others'].outputs[0]].val
    assert output.sizes==[64,262144]
    manifests={}
    for name in forwards:
        for delegate in methods[name].delegates:
            for compile_spec in delegate.compile_specs:
                if compile_spec.key!='offgraph_kv_manifest':continue
                manifest=json.loads(bytes(compile_spec.value))
                assert manifest['layout']=='batched-sequence' and manifest['maximum_capacity']==args.max_context
                assert manifest['max_seqs']==args.max_kv_seqs and manifest['max_segments']==2*args.max_kv_seqs
                assert manifest['max_write']==args.max_step_tokens and manifest['causal'] is True
                assert not manifest.get('context_window',False)
                rings=[layer for layer in manifest['layers'] if layer['policy']=='ring']
                assert len(manifest['layers'])==48 and len(rings)==40
                assert all(layer['window']==1024 for layer in rings)
                manifests[name]=manifest
    assert set(manifests)==forwards
    builds=[json.loads(line) for line in (args.job_dir/'actual-compiler-optimization.jsonl').read_text().splitlines()]
    optimized=[b for b in builds if b['optimization_flags'] and b['optimization_flags'][-1]=='-O2']
    assert optimized and not any(b['optimization_flags'] and b['optimization_flags'][-1]=='-O0' for b in builds)
    columns=args.max_kv_seqs+1
    expected_kv=40*(1024+args.max_step_tokens-1)*columns*8*256*4+8*args.max_context*columns*1*512*4
    policy=json.loads((args.output/'sampler-policy.json').read_text())
    assert sorted(policy['stop_tokens'])==[1,50,106] and policy['suppress_tokens']==[258882,258883]
    audit=dict(fresh_export=True,selected_output_bound=output.sizes,constant_metadata=metadata,
               offgraph_manifests=manifests,GGUF_sha256=digest(args.gguf),
               packer_sha256=digest(Path(cuda_packers.__file__)),precision_kernel_sha256=digest(args.precision_control),
               artifacts={name:dict(sha256=digest(args.output/name),bytes=(args.output/name).stat().st_size)
                          for name in ['model.pte','aoti_cuda_blob.ptd']},
               expected_KV_bytes=expected_kv,expected_KV_GiB=expected_kv/2**30,
               AOTI_wrapper_optimization='O2',actual_optimized_builds=len(optimized),
               sampler_policy=policy,
               prefill_activation='BF16; native small tails retained',physical_columns=columns,
               max_resident_sequences=args.max_kv_seqs,
               supported_prefix_entries_with_16_active=max(0,args.max_kv_seqs-16-1),
               context=args.max_context,active_sessions=16,decode_ceiling=16,static_methods=[1,2,4,8,16,32,64],
               safe_ring_rows=1024+args.max_step_tokens-1,runtime_and_source_checkout_unchanged=True,
               same_GGUF_and_packer=True,PTD_binary_identity_with_previous_export_not_assumed=True,
               deployment_accepted=False,performance_GMRID_and_sampled_memory_acceptance_pending=True)
    (args.output/'artifact-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    state('artifact-exported-and-audited',expected_KV_GiB=expected_kv/2**30,
          performance_and_GMRID_acceptance_pending=True)


if __name__=='__main__':
    main()
