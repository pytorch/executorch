"""Bounded real Gemma GGUF A16 prefill test with FP32 decode metadata.

Four configurations per Gate/Down M128/512, four paired AB/BA rounds,
discard first. Timings include preparing scale/zero on every invocation.
No model route changes, no A100 changes, no permanent metadata cache.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while raw := f.read(8 * 1024 * 1024): h.update(raw)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=900)
    a = p.parse_args(); a.output.mkdir(exist_ok=False)
    report = dict(scope=__doc__, controller_pid=os.getpid(), cases=[],
                  A16=True, weight_encoding_changed=False, new_KA=False,
                  model_dispatch_changed=False, full_model_or_HTTP_accepted=False)
    def state(status, **extra):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        (a.output / 'progress.json').write_text(json.dumps(dict(
            status=status, controller_pid=os.getpid(), **extra), indent=2) + '\n')
    try:
        state('waiting-GPU-lock')
        with a.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip()
            started = time.monotonic()
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor-cache'),
                TRITON_CACHE_DIR=str(a.output / 'triton-cache'),
                TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', OMP_NUM_THREADS='2')
            sys.path.insert(0, str(a.et_source.parent))
            import numpy as np
            import torch
            from gguf import GGUFReader, GGMLQuantizationType
            from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
            from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import chunked_dequant_linear
            from executorch.extension.llm.export.quant import to_default, fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda

            from executorch.backends.cuda.autotune import cuda_graph_timing as clock

            import public_prefill_gemma_fp32 as candidate
            assert torch.cuda.get_device_capability() == (12, 0)
            assert sha(a.gguf) == '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
            torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
            report.update(GGUF_SHA256=sha(a.gguf), candidate_SHA256=sha(candidate.__file__),
                device=torch.cuda.get_device_name(), torch=torch.__version__,
                compile_options=dict(max_autotune_gemm=True,
                    max_autotune_gemm_backends='TRITON', emulate_precision_casts=True,
                    **{'triton.cudagraphs': False}))
            reader = GGUFReader(str(a.gguf))
            def portable(t):
                kind = {GGMLQuantizationType.Q4_0:'q4_0', GGMLQuantizationType.Q4_1:'q4_1'}[t.tensor_type]
                shape = tuple(map(int, t.shape[::-1]))
                raw = torch.from_numpy(np.array(t.data, copy=True)).view(torch.uint8).reshape(-1)
                return decode_q4_blocks(raw, kind, shape)
            def reference_fp32(x, qdata, scale, scale_step, zero, zero_step, group_size):
                # Exact arithmetic of the selected Gemma precision export.
                n, khalf = qdata.shape; k = khalf*2; groups = k//group_size; gps=256//group_size
                def rows(i,j):
                    packed=qdata[i:j].reshape(j-i,groups,group_size//2)
                    q=torch.stack(((packed&15).float(),(packed>>4).float()),dim=-1).reshape(j-i,groups,group_size)
                    s=scale[i:j].float()*scale_step[i:j].float().repeat_interleave(gps,dim=1)
                    z=zero[i:j].float()*zero_step[i:j].float().repeat_interleave(gps,dim=1)
                    dense=((q-z[:,:,None])*s[:,:,None]).reshape(j-i,k).to(x.dtype)
                    return torch.nn.functional.linear(x,dense)
                return chunked_dequant_linear(x,n,rows)
            def paired(ref, own):
                samples = dict(reference=[], candidate=[])
                for r in range(4):
                    order = [('reference', ref), ('candidate', own)]
                    if r % 2: order.reverse()
                    for name, fn in order:
                        assert time.monotonic() - started < a.max_seconds
                        for _ in range(3): fn()
                        samples[name].append(clock.time_with_cuda_graph_us(fn))
                means = {k: statistics.mean(v[1:]) for k, v in samples.items()}
                return dict(samples_us=samples, mean_last3_us=means,
                    sample_SD_us={k: statistics.stdev(v[1:]) for k, v in samples.items()},
                    candidate_over_reference=means['candidate'] / means['reference'])
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream):
                for role in ['ffn_gate_up', 'ffn_down']:
                    if role == 'ffn_gate_up':
                        names = ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight']
                        t = fuse_along_output([portable(next(t for t in reader.tensors if t.name == name)) for name in names])
                        n, k = t.shape
                    else:
                        raw = next(t for t in reader.tensors if t.name == 'blk.0.ffn_down.weight')
                        names = [raw.name]; t = portable(raw); n, k = map(int, raw.shape[::-1])
                    module = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                    pack_linear_for_cuda(module, {'weight': t})
                    w = tuple(getattr(module.weight, key).to('cuda').contiguous()
                        for key in ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                    sg = torch.empty((n, k // 32), device='cuda', dtype=torch.float32)
                    zg = torch.empty_like(sg)
                    q = torch.stack(((w[0]&15).float(),(w[0]>>4).float()),dim=-1).reshape(n,k//32,32)
                    scale=w[1].float()*w[2].float().repeat_interleave(8,dim=1)
                    zero=w[3].float()*w[4].float().repeat_interleave(8,dim=1)
                    dense=((q-zero[:,:,None])*scale[:,:,None]).reshape(n,k).bfloat16()
                    del q,scale,zero
                    prepared_dense = candidate.materialize(w, sg, zg)
                    torch.cuda.synchronize()
                    assert torch.equal(dense, prepared_dense), 'Prepared metadata changed effective weights'
                    del prepared_dense
                    weight = dict(GGUF_tensors=names, packed_SHA256=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in w],
                        verified_effective_weight_elements=n*k, effective_weight_different_elements=0,
                        temporary_metadata_bytes=sg.numel()*sg.element_size()+zg.numel()*zg.element_size(),
                        persistent_extra_model_bytes=0, arithmetic='FP32-decode-then-BF16-GEMM', source_quant_types=[next(t for t in reader.tensors if t.name==name).tensor_type.name for name in names])
                    ref = torch.compile(reference_fp32, fullgraph=True, dynamic=False,
                        options=report['compile_options'])
                    example = torch.randn((512, k), device='cuda', dtype=torch.bfloat16)
                    torch._dynamo.mark_dynamic(example, 0, min=128, max=512)
                    state('compiling-original-reference', role=role)
                    ref(example, *w, 32); torch.cuda.synchronize(); del example
                    for m in [128, 512]:
                        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16,
                            generator=torch.Generator(device='cuda').manual_seed(47000+n+m))
                        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
                        target = torch.nn.functional.linear(x.float(), dense.float()).bfloat16()
                        reference = lambda: ref(x, *w, 32)
                        for config in candidate.configs():
                            state('testing-prepared-groups', role=role, M=m, config=asdict(config))
                            own = lambda: candidate.launch(x, w, sg, zg, out, config)
                            case = dict(role=role, M=m, N=n, K=k, weight=weight, config=asdict(config))
                            try:
                                ys = [own().clone() for _ in range(3)]; torch.cuda.synchronize()
                                assert all(torch.equal(ys[0], y) for y in ys[1:])
                                delta = ys[0].float()-target.float()
                                rms = float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-8))
                                assert torch.isfinite(ys[0]).all() and rms < .005
                                torch.testing.assert_close(ys[0].float(), target.float(), rtol=.05, atol=4.)
                                case.update(numeric_pass=True, deterministic=True, relative_RMS=rms,
                                    timing=paired(reference, own), timing_includes_metadata_preparation=True)
                                del ys, delta
                            except Exception as e:
                                case.update(numeric_pass=False, error=str(e))
                            report['cases'].append(case)
                        del x, out, target
                    del w, sg, zg, dense, module, t, ref
                    torch._dynamo.reset(); torch.cuda.empty_cache()
            state('completed', cases=len(report['cases']))
    except Exception as e:
        report['traceback'] = traceback.format_exc()
        state('failed', error=str(e)); raise


if __name__ == '__main__': main()
