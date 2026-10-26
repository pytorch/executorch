"""Compare unchanged attention math with global and independent KV addressing.

This tests the native kernels at production launch geometry. It does not
replace the mini model, HTTP or full-model performance qualification.
"""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--full-owner-padding', type=int, choices=[0,1,3,8], default=0)
    parser.add_argument('--gemma-production', action='store_true',
                        help='Preserve qualified Gemma kernels and use its actual D256/D512 head counts')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    old = json.loads((package / 'runtime_kv_columns_source.json').read_text())
    new = json.loads((package / 'independent_session_capacity_source.json').read_text())
    name = 'backends/cuda/triton/kernels/sdpa_varlen.py'
    assert not args.output.exists(), 'Preserve previous attempts'
    args.output.mkdir(parents=True)
    kernel_root = args.output / 'source'
    file = kernel_root / name
    file.parent.mkdir(parents=True)
    progress = args.output / 'progress.json'

    def state(status, **fields):
        value = dict(status=status, pid=os.getpid(),
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     full_model_qualified=False, **fields)
        pending = progress.with_suffix('.pending.json')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(progress)

    try:
        state('preparing-exact-public-kernel')
        available = subprocess.run(['git', '-C', str(args.public_source),
                                   'cat-file', '-e', old['base_commit'] + ':' + name],
                                  capture_output=True)
        if available.returncode:
            subprocess.run(['git', '-C', str(args.public_source), 'fetch', '--depth=1',
                            'origin', old['base_commit']], check=True, timeout=300)
        file.write_bytes(subprocess.check_output(['git', '-C', str(args.public_source),
                         'show', old['base_commit'] + ':' + name]))
        assert digest(file) == old['changed_files'][name]['before_sha256']
        patches = ['runtime_kv_columns.patch', 'independent_session_capacity.patch']
        if args.gemma_production:
            new = json.loads((package / 'gemma_independent_session_attention.json').read_text())
            patches = ['runtime_kv_columns.patch', 'gemma_runtime_columns_overlay.patch',
                       'gemma_independent_session_attention.patch']
            assert digest(package / patches[-1]) == new['patch_SHA256']
        for patch in patches:
            subprocess.run(['git', 'apply', '--include=' + name, str(package / patch)],
                           cwd=kernel_root, check=True)
        assert digest(file) == new['files'][name]['after_SHA256']
        os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(args.output / 'inductor'),
                          TRITON_CACHE_DIR=str(args.output / 'triton'),
                          TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                          PYTHONDONTWRITEBYTECODE='1')
        import torch
        import triton
        spec = importlib.util.spec_from_file_location('independent_attention', file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(2)

        def launch(q, k, v, info, ring):
            _, hq, tokens, dim = q.shape
            hkv, groups = k.shape[2], hq // k.shape[2]
            logical_rows = ring or 131072
            decode = tokens <= (16 if dim in (128, 512) else 4)
            splits = max(1, min(16, triton.cdiv(logical_rows, 256))) if decode else 1
            if args.gemma_production:
                decode = tokens <= (64 if dim == 512 else 4)
                splits = max(1, min(16, triton.cdiv(logical_rows, 256))) if decode else 1
                if dim == 512 and decode:
                    splits = max(1, min(128, triton.cdiv(logical_rows, 128),
                                        triton.cdiv(256, tokens * hkv)))
            out = torch.empty_like(q) if decode else torch.zeros_like(q)
            if splits > 1:
                partial = torch.empty((splits,hq,tokens,dim), device='cuda', dtype=torch.float32)
                m = torch.empty((splits,hq,tokens), device='cuda', dtype=torch.float32)
                l = torch.empty_like(m)
            else:
                partial = m = l = out
            common = [q,k,v,out,partial,m,l,info,hq,tokens,
                      q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2),
                      v.stride(0),v.stride(1),v.stride(2),out.stride(1),out.stride(2),
                      1/math.sqrt(dim),logical_rows]
            kwargs = dict(INFO_WIDTH=info.shape[1], NUM_SPLITS=splits,
                          IS_CAUSAL=True, CONTEXT_WINDOW=False,
                          WINDOW=(1024 if args.gemma_production else 128) if ring else 0, RING=ring,
                          HEAD_DIM=dim, NUM_GROUPS=groups,
                          num_warps=8 if dim==512 else 4)
            if decode:
                module._sdpa_varlen_decode_kernel[(tokens,hkv,splits)](
                    *common, **kwargs, NUM_SEGMENTS=info.shape[0],
                    BLOCK_N=32 if dim==512 else (64 if args.gemma_production else 128),
                    BLOCK_G=triton.next_power_of_2(groups), num_stages=1 if dim==512 or args.gemma_production else 2)
            else:
                block_m = 16 if dim==512 or tokens*groups<=16 else 64
                num_qb = triton.cdiv(tokens*groups,block_m)
                unpack = args.gemma_production and dim == 256 and tokens > 64
                if args.gemma_production and dim == 256:
                    block_m = 32
                num_qb = triton.cdiv(tokens * (1 if unpack else groups), block_m)
                grid_tiles = num_qb + info.shape[0] - 1 if args.gemma_production else info.shape[0]*num_qb
                generic_kwargs = {**kwargs, 'NUM_GROUPS': 1 if unpack else groups}
                if args.gemma_production:
                    generic_kwargs.update(NUM_SEGMENTS=info.shape[0], COMPACT_GRID=True,
                                          KV_HEAD_GROUPS=groups if unpack else 1)
                module._sdpa_varlen_kernel[(grid_tiles,hq if unpack else hkv,splits)](
                    *common,num_qb, **generic_kwargs, BLOCK_M=block_m,
                    BLOCK_N=32 if dim==512 or args.gemma_production else 64,
                    num_stages=1 if dim==512 else (2 if args.gemma_production else (3 if tokens<=64 else 2)))
            if splits > 1:
                module._sdpa_varlen_reduce_kernel[(hq*tokens,)](
                    partial,m,l,out,hq,tokens,out.stride(1),out.stride(2),
                    NUM_SPLITS=splits,BLOCK_S=triton.next_power_of_2(splits),
                    HEAD_DIM=dim)
            return out

        results = []
        cases = [('decode1',[1024],[1],1),
                 ('decode2',[7,1024],[1,1],2),
                 ('decode4',[513,1024,2047,4097],[1]*4,4),
                 ('decode8',[7,65,513,1024,4097,16,1024,2047],[1]*8,8),
                 ('decode16',[7,65,513,1024,4097,16,1024,2047]*2,[1]*16,16),
                 ('prefill64',[1024],[64],64),
                 ('prefill512',[1024],[512],512),
                 ('decode8-prefill64',[1024]*9,[1]*8+[64],128),
                 ('decode8-prefill512',[1024]*9,[1]*8+[512],520)]
        if args.gemma_production:
            cases[-2] = ('decode8-prefill64',[1024]*9,[1]*8+[64],72)
            cases[-1] = ('decode8-prefill504',[1024]*9,[1]*8+[504],512)
        geometries = [(256,16,8),(512,16,1)] if args.gemma_production else [(128,32,2),(512,8,1)]
        with args.lock.open('a') as lock:
            state('waiting-GPU-lock')
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                               '--format=csv,noheader'],text=True).strip()
            assert torch.cuda.get_device_capability() == (12,0)
            for dim,hq,hkv in geometries:
                for label,histories,widths,tokens in cases:
                    rings = [1024+512-1] if args.gemma_production and dim == 256 else (
                        [0] if args.gemma_production else [0,128+512-1])
                    window = 1024 if args.gemma_production else 128
                    for ring in rings:
                        state('running-kernel-oracle', case=label, dimension=dim, ring=ring)
                        torch.manual_seed(4106+tokens+dim+ring)
                        owners = len(histories)
                        capacities = [max(512,2**math.ceil(math.log2(h+w)))
                                      for h,w in zip(histories,widths)]
                        rows = ring or max(capacities)
                        cols = owners+1
                        total = sum(capacities)+owners*args.full_owner_padding+1
                        bases = [sum(capacities[:i])+i*args.full_owner_padding for i in range(owners)]
                        global_k = torch.full((rows,cols,hkv,dim),float('nan'),device='cuda',dtype=torch.bfloat16)
                        global_v = torch.full_like(global_k,float('nan'))
                        packed_k = torch.full((total,1,hkv,dim),float('nan'),device='cuda',dtype=torch.bfloat16) if not ring else global_k.clone()
                        packed_v = torch.full_like(packed_k,float('nan')) if not ring else global_v.clone()
                        q = torch.randn((1,hq,tokens,dim),device='cuda',dtype=torch.bfloat16)
                        segments = 36 if args.gemma_production else 42
                        info5 = torch.zeros((segments,5),device='cuda',dtype=torch.int32)
                        info7 = torch.zeros((segments,7),device='cuda',dtype=torch.int32)
                        info5[:,4] = cols
                        info7[:,4] = cols
                        reference = torch.zeros_like(q)
                        start = 0
                        for owner,(history,width) in enumerate(zip(histories,widths)):
                            end = history+width
                            source_k = torch.randn((end,hkv,dim),device='cuda',dtype=torch.bfloat16)
                            source_v = torch.randn_like(source_k)
                            if ring:
                                logical = torch.arange(max(0,end-ring),end,device='cuda')
                                slots = logical%ring
                                global_k[slots,owner] = source_k[logical]
                                global_v[slots,owner] = source_v[logical]
                                packed_k[slots,owner] = source_k[logical]
                                packed_v[slots,owner] = source_v[logical]
                            else:
                                global_k[:end,owner] = source_k
                                global_v[:end,owner] = source_v
                                packed_k[bases[owner]:bases[owner]+end,0] = source_k
                                packed_v[bases[owner]:bases[owner]+end,0] = source_v
                            info5[owner] = torch.tensor([start,width,owner,end,cols],device='cuda',dtype=torch.int32)
                            info7[owner] = torch.tensor([start,width,owner,end,cols,bases[owner],capacities[owner]],device='cuda',dtype=torch.int32)
                            queries = q[0,:,start:start+width].float()
                            k_ref = source_k.repeat_interleave(hq//hkv,dim=1).permute(1,0,2).float()
                            v_ref = source_v.repeat_interleave(hq//hkv,dim=1).permute(1,0,2).float()
                            scores = queries @ k_ref.transpose(1,2) / math.sqrt(dim)
                            kp = torch.arange(end,device='cuda')[None,:]
                            qp = torch.arange(history,end,device='cuda')[:,None]
                            visible = kp<=qp
                            if ring:
                                visible &= kp>qp-window
                            scores.masked_fill_(~visible,float('-inf'))
                            reference[0,:,start:start+width] = (scores.softmax(-1) @ v_ref).to(torch.bfloat16)
                            start += width
                        arrays = [(global_k,global_v,info5),(packed_k,packed_v,info7)]
                        errors = []
                        graphs = []
                        outputs = []
                        for k,v,info in arrays:
                            result = launch(q,k,v,info,ring)
                            torch.testing.assert_close(result,reference,atol=.015,rtol=.02)
                            relative = float(torch.linalg.vector_norm((result-reference).float()) /
                                             torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
                            assert relative < .008, relative
                            errors.append({'max_abs':float((result-reference).abs().max()),'relative_L2':relative})
                            for _ in range(5):
                                launch(q,k,v,info,ring)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                output = launch(q,k,v,info,ring)
                            graphs.append(graph)
                            outputs.append(output)
                        rounds = [[],[]]
                        for round_id in range(4):
                            for index in ([0,1] if round_id%2==0 else [1,0]):
                                begin,finish = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                begin.record()
                                for _ in range(30):
                                    graphs[index].replay()
                                finish.record()
                                finish.synchronize()
                                rounds[index].append(begin.elapsed_time(finish)/30)
                                torch.testing.assert_close(outputs[index],reference,atol=.015,rtol=.02)
                        means = [statistics.mean(r[1:]) for r in rounds]
                        results.append(dict(case=label,dimension=dim,query_heads=hq,
                                            KV_heads=hkv,ring=ring,owners=owners,
                                            query_tokens=tokens,active_query_tokens=sum(widths),
                                            histories=histories,capacities=capacities,
                                            global_full_slots=rows*cols if not ring else None,
                                            independent_full_slots=total if not ring else None,
                                            full_owner_padding_rows=args.full_owner_padding,
                                            logical_full_launch_rows=131072,
                                            oracle_errors=errors,rounds_ms=rounds,means_ms=means,
                                            sample_SD_ms=[statistics.stdev(r[1:]) for r in rounds],
                                            independent_over_global=means[1]/means[0]))
                        (args.output/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
                        del graphs,outputs,arrays,global_k,global_v,packed_k,packed_v,q,reference
                        torch.cuda.empty_cache()
        (args.output/'results.json').write_text(json.dumps(dict(
            kernel_SHA256=digest(file),source_spec_SHA256=digest(package/(
                'gemma_independent_session_attention.json' if args.gemma_production else 'independent_session_capacity_source.json')),
            oracle='FP32 causal per-owner attention over original BF16 K/V',
            production_launch_geometry=True,gemma_production_geometries=args.gemma_production,
            all_cases_pass=True,cells=results,
            mini_model_qualified=False,full_models_qualified=False),indent=2)+'\n')
        state('completed',cells=len(results),all_oracles_pass=True)
    except Exception as error:
        state('failed',detail=str(error)[-2000:],traceback=traceback.format_exc()[-2500:])
        raise


if __name__ == '__main__':
    main()
