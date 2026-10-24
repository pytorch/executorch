"""Read-only native decoder and independent GEMM arithmetic diagnostic.

This materializes the existing JIT decoder, without editing it, and records
BF16 differences versus explicit staged eager arithmetic. No tolerance change
or automatic kernel promotion is performed.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def materialize(q,sc,ss,z,zs,out,N:tl.constexpr,K:tl.constexpr,
                stride_qn:tl.constexpr,stride_qk:tl.constexpr,
                stride_sn:tl.constexpr,stride_sk:tl.constexpr,
                stride_ssn:tl.constexpr,stride_ssk:tl.constexpr,
                stride_zn:tl.constexpr,stride_zk:tl.constexpr,
                stride_zsn:tl.constexpr,stride_zsk:tl.constexpr,
                decode:tl.constexpr,BLOCK_N:tl.constexpr):
    ns=tl.program_id(0)*BLOCK_N+tl.arange(0,BLOCK_N)
    sb=tl.program_id(1)
    weight=decode(q,sc,ss,z,zs,ns,ns<N,sb,
        stride_qn,stride_qk,stride_sn,stride_sk,stride_ssn,stride_ssk,
        stride_zn,stride_zk,stride_zsn,stride_zsk,BLOCK_N)
    ks=sb*256+tl.arange(0,256)
    tl.store(out+ns[:,None]*K+ks[None,:],weight,mask=ns[:,None]<N)

def diagnose(native,x,w,reference,output):
    n,k=w[0].shape[0],x.shape[1]
    q,sc,ss,z,zs=w
    def stats(a,b):
        d=a.float()-b.float()
        return dict(relative_rms=float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8)),
                    max_abs=float(d.abs().max()),different_elements=int((a!=b).sum()),
                    finite=bool(torch.isfinite(a).all()),diagnostic_only=True)
    staged=[]
    for start in range(0,n,256):
        end=min(start+256,n)
        qs=torch.stack((q[start:end]&15,q[start:end]>>4),-1).bfloat16().reshape(end-start,k//32,32)
        sg=(sc[start:end].bfloat16()*ss[start:end].bfloat16().repeat_interleave(8,1)).bfloat16()
        zg=(z[start:end].bfloat16()*zs[start:end].bfloat16().repeat_interleave(8,1)).bfloat16()
        staged.append(((qs-zg[...,None]).bfloat16()*sg[...,None]).bfloat16().reshape(end-start,k))
    staged=torch.cat(staged,0)
    y=reference();again=reference();assert torch.equal(y,again)
    result=dict(native_output_deterministic=True,N=n,K=k,M=x.shape[0],dtype=str(x.dtype),
                source_native_decoder_unchanged=True,threshold_unchanged=.005,
                native_vs_eager_staged=stats(y,torch.nn.functional.linear(x.float(),staged.float()).bfloat16()),controls=[])
    for fusion in [True,False]:
        dense=torch.empty((n,k),device=x.device,dtype=torch.bfloat16)
        compiled=materialize[(triton.cdiv(n,32),k//256)](q,sc,ss,z,zs,dense,N=n,K=k,
            stride_qn=q.stride(0),stride_qk=q.stride(1),stride_sn=sc.stride(0),stride_sk=sc.stride(1),
            stride_ssn=ss.stride(0),stride_ssk=ss.stride(1),stride_zn=z.stride(0),stride_zk=z.stride(1),
            stride_zsn=zs.stride(0),stride_zsk=zs.stride(1),decode=native._decode_w4a16_superblock,BLOCK_N=32,
            num_warps=4,num_stages=1,enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        pointwise=stats(dense,staged)
        gemm=torch.nn.functional.linear(x.float(),dense.float()).bfloat16()
        mismatches=(dense!=staged).nonzero()[:16].cpu().tolist()
        examples=[dict(n=i,k=j,jit=float(dense[i,j]),staged=float(staged[i,j])) for i,j in mismatches]
        path=output/('native-decoder-fusion-'+str(fusion)+'.ptx');path.write_text(compiled.asm['ptx'])
        result['controls'].append(dict(enable_fp_fusion=fusion,decoder_vs_eager_staged=pointwise,
            native_GEMM_vs_materialized_decoder_FP32_linear=stats(y,gemm),mismatch_examples=examples,
            PTX_file=path.name))
        del dense,gemm
    return result
