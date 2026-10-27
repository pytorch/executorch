"""Compare MG two-source attention reductions to the exact original SDK kernel.

Synthetic identical BF16 Q/K/V, no models or PTEs. This diagnoses reduction
configuration only; it cannot qualify full-model logits or exported graphs.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import torch
import triton
import triton.testing
from torch.library import wrap_triton

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def aligned_decode(mod,q,k,v,kn,vn,info,window,scale,splits):
    heads,total,dim=q.shape[1:];kvheads=kn.shape[1];groups=heads//kvheads
    out=torch.empty_like(q)
    partial=torch.empty((splits,heads,total,dim),device=q.device,dtype=torch.float32)
    maximum=torch.empty((splits,heads,total),device=q.device,dtype=torch.float32)
    normalizer=torch.empty_like(maximum)
    wrap_triton(mod._decode)[(total,kvheads,splits)](
        q,k,v,kn,vn,kn.stride(1),kn.stride(2),vn.stride(1),vn.stride(2),
        out,partial,maximum,normalizer,info,heads,total,
        q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2),v.stride(0),v.stride(1),v.stride(2),out.stride(1),out.stride(2),
        scale,window,info.shape[0],info.shape[1],splits,True,False,window,window,128,dim,groups,triton.next_power_of_2(groups),num_warps=4,num_stages=2)
    wrap_triton(mod._reduce)[(heads*total,)](partial,maximum,normalizer,out,heads,total,out.stride(1),out.stride(2),splits,triton.next_power_of_2(splits),dim,num_warps=4)
    return out

def compare(a,b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=a.double()-b.double();denom=a.double().square().mean().sqrt().clamp_min(1e-12)
    return dict(normalized_RMS=float(delta.square().mean().sqrt()/denom),max_abs=float(delta.abs().max()),byte_equal=bool(torch.equal(a,b)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-kernel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(args.baseline_kernel.read_bytes()).hexdigest()=='727df52b527420362bade0a9dc1f638e8c13850d7035e006e27cff8659d40df8'
    candidate=Path(__file__).with_name('two_source_ring.py')
    baseline=load('original_reduction_reference',args.baseline_kernel)
    mod=load('two_source_reduction_candidate',candidate)
    rows=[]
    with args.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        torch.manual_seed(310027);torch.backends.cuda.matmul.allow_tf32=False
        window=2048;budget=4096;capacity=window+budget;heads=32;kvheads=2;dim=128;scale=dim**-.5
        for width,count in [(1,1),(1,2),(1,4),(1,8),(1,16),(64,1),(512,1)]:
            total=width*count;columns=count+1
            info=torch.zeros((42,7),device='cuda',dtype=torch.int32)
            oldk=torch.zeros((capacity,columns,kvheads,dim),device='cuda',dtype=torch.bfloat16);oldv=torch.zeros_like(oldk)
            hk=torch.zeros((window,columns,kvheads,dim),device='cuda',dtype=torch.bfloat16);hv=torch.zeros_like(hk)
            kn=torch.empty((1,kvheads,total,dim),device='cuda',dtype=torch.bfloat16);vn=torch.empty_like(kn)
            q=torch.randn((1,heads,total,dim),device='cuda',dtype=torch.bfloat16)
            for owner in range(count):
                length=1041+owner*17;start=owner*width
                ks=torch.randn((length+width,kvheads,dim),device='cuda',dtype=torch.bfloat16);vs=torch.randn_like(ks)
                oldk[:length+width,owner]=ks;oldv[:length+width,owner]=vs
                hk[:length,owner]=ks[:length];hv[:length,owner]=vs[:length]
                kn[0,:,start:start+width]=ks[length:].transpose(0,1);vn[0,:,start:start+width]=vs[length:].transpose(0,1)
                info[owner]=torch.tensor([start,width,owner,length+width,columns,0,capacity],device='cuda',dtype=torch.int32)
            funcs={'baseline':lambda:baseline.sdpa_varlen(q,oldk,oldv,info,scale,window=window,ring=capacity,rows_available=capacity),
                   'native-two-source':lambda:mod.sdpa_ring_current(q,hk,hv,kn,vn,info,window,scale)}
            if width==1:funcs['aligned-16-splits']=lambda:aligned_decode(mod,q,hk,hv,kn,vn,info,window,scale,16)
            else:funcs['aligned-BN128']=lambda:mod.sdpa_ring_current(q,hk,hv,kn,vn,info,window,scale,block_n=128)
            outputs={name:f().clone() for name,f in funcs.items()};torch.cuda.synchronize()
            numeric={name:compare(outputs['baseline'],value) for name,value in outputs.items() if name!='baseline'}
            timing={name:[] for name in funcs}
            for number in range(4):
                for name in (list(funcs) if number%2==0 else list(funcs)[::-1]):
                    timing[name].append(float(triton.testing.do_bench_cudagraph(funcs[name],rep=60)))
            row=dict(width=width,owners=count,total_tokens=total,numerics=numeric,
                     timing={name:dict(mean_ms=statistics.mean(v[1:]),sample_SD_ms=statistics.stdev(v[1:]),rounds=v) for name,v in timing.items()})
            rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        result=dict(rows=rows,baseline_source_SHA256=hashlib.sha256(args.baseline_kernel.read_bytes()).hexdigest(),candidate_source_SHA256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    synthetic_only=True,model_logits_qualified=False,exported_dynamic_shape_qualified=False,
                    torch=torch.__version__,triton=triton.__version__,GPU=torch.cuda.get_device_name(),timing_mode='CUDA graph replay excludes Python submission gaps')
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
