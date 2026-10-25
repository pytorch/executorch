"""Screen public grouped W4A16 against unchanged ET using actual GGUF packing.

Standalone screen only: no model dispatch changes, KernelAgent invocation or
automatic promotion. Native M8/16/64 and dynamic bulk entries use their own
independently checked arithmetic. Standalone Inductor timings are not the
current model's AOTI launch configuration or HTTP performance.
"""
import argparse
from dataclasses import asdict
import datetime
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--roles', nargs='+', choices=['ffn_gate_up', 'ffn_down'],
                   default=['ffn_gate_up', 'ffn_down'])
    p.add_argument('--entries', nargs='+', default=['native8', 'native16', 'native64',
                   'bulk64', 'bulk72', 'bulk128', 'bulk512', 'bulk2048'])
    p.add_argument('--max-seconds', type=int, default=5400)
    p.add_argument('--packed32', action='store_true', help='Read identical quantized bytes as uint32 words; standalone only')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, controller_pid=os.getpid(), entries=a.entries,
                  results=[], weight_sources={}, materialization_checks=[],
                  rounds=4, discard=[1], mean_of=[2, 3, 4], numeric_gate=.005,
                  activation='Synthetic BF16; fixed seeded inputs',
                  unchanged_native_route=True, automatic_promotion=False,
                  KA_invoked=False, A100_performance_accepted=False,
                  actual_model_AOTI_and_full_server_acceptance_pending=True)
    def save(status, **fields):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')
    save('waiting-GPU-lock')
    a.lock.parent.mkdir(parents=True, exist_ok=True)
    held_lock = a.lock.open('a+')
    fcntl.flock(held_lock, fcntl.LOCK_EX)
    started = time.monotonic()
    def bounded():
        assert time.monotonic() - started < a.max_seconds, 'Standalone screening time limit'
        owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).splitlines()
        assert all(int(v) == os.getpid() for v in owners if v.strip()), 'Other GPU owner present'
    try:
        bounded()
        os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor-cache'),
                          TRITON_CACHE_DIR=str(a.output / 'triton-cache'),
                          TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', OMP_NUM_THREADS='2')
        sys.path.insert(0, str(a.et_source.parent))
        import numpy as np
        import torch
        import triton
        from gguf import GGUFReader, GGMLQuantizationType
        from torch._library.triton import set_wrap_triton_enabled
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor
        from executorch.extension.llm.export.quant import to_default, fuse_along_output
        from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
        from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
        from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
        from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
        from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
        from executorch.backends.cuda.autotune import cuda_graph_timing as clock
        from native_gemm_oracle_diagnostic import native_fma_dense
        if a.packed32:
            import public_grouped_w4a16_packed32 as candidate
        else:
            import public_grouped_w4a16 as candidate
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        assert torch.cuda.get_device_capability() == (12, 0), 'Screen is explicitly SM120'
        assert digest(native.__file__) == 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
        assert digest(inspect.getfile(_dequant_matmul)) == '1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'
        assert digest(inspect.getfile(pack_linear_for_cuda)) == 'c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
        assert digest(inspect.getfile(to_default)) == '5daca6d9330b9b32a75dbcc8076a8c17dc433402509a87dff06a27ea1f22eeb5'
        assert digest(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
        report.update(device=torch.cuda.get_device_name(), capability=[12, 0],
                      torch=torch.__version__, Triton=triton.__version__,
                      GGUF_SHA256=digest(a.gguf), candidate_SHA256=digest(candidate.__file__),
                      native_source_SHA256=digest(native.__file__),
                      bulk_baseline='Unmodified ET _dequant_matmul, isolated Inductor dynamic-M compilation',
                      timing='Existing native cold-L2 cuda_graph_timing.time_with_cuda_graph_us; four paired AB/BA rounds',
                      standalone_not_current_model_AOTI=True)
        report['uint32_view_of_original_bytes'] = a.packed32
        options = dict(max_autotune_gemm=True, max_autotune_gemm_backends='TRITON',
                       emulate_precision_casts=True, **{'triton.cudagraphs': False})
        report['standalone_bulk_compile_options'] = options
        save('reading-actual-GGUF')
        reader = GGUFReader(str(a.gguf))
        def portable(tensor):
            assert tensor.tensor_type == GGMLQuantizationType.Q4_K
            n, k = tuple(map(int, tensor.shape[::-1]))
            raw = torch.from_numpy(np.array(tensor.data, copy=True)).view(torch.uint8).reshape(n, -1)
            report['weight_sources'][tensor.name] = dict(quant='Q4_K', N=n, K=k,
                original_raw_SHA256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())
            return to_default(tensor.name, ExportableGGUFTensor.from_raw(raw, 'q4_k', torch.bfloat16))
        def staged_dense(w, start, end):
            q, sc, ss, z, zs = w
            k = q.shape[1] * 2
            codes = torch.stack((q[start:end] & 15, q[start:end] >> 4), -1).bfloat16().reshape(end - start, k // 32, 32)
            sg = (sc[start:end].bfloat16() * ss[start:end].bfloat16().repeat_interleave(8, 1)).bfloat16()
            zg = (z[start:end].bfloat16() * zs[start:end].bfloat16().repeat_interleave(8, 1)).bfloat16()
            return ((codes - zg[:, :, None]).bfloat16() * sg[:, :, None]).bfloat16().reshape(end - start, k)
        def oracle(x, w, arithmetic):
            dense = native_fma_dense if arithmetic == 'native-zero-fma-bf16' else staged_dense
            x_fp32 = x.float()
            return torch.cat([torch.nn.functional.linear(x_fp32, dense(w, i, min(i + 256, w[0].shape[0])).float()).bfloat16()
                for i in range(0, w[0].shape[0], 256)], 1)
        def checked(fn, reference):
            # Candidate timings use a preallocated output. Clone here so the
            # deterministic check observes four independent launch results.
            ys = [fn().clone() for _ in range(4)]
            torch.cuda.synchronize()
            assert all(torch.equal(ys[1], y) for y in ys[2:]), 'Nondeterministic output'
            y = ys[1]
            assert y.shape == reference.shape and y.dtype == torch.bfloat16
            delta = y.float() - reference.float()
            rms = float(delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-8))
            assert bool(torch.isfinite(y).all()) and rms < .005, f'Unchanged RMS gate: {rms}'
            torch.testing.assert_close(y.float(), reference.float(), rtol=.05, atol=4.0)
            return dict(relative_RMS=rms, max_abs=float(delta.abs().max()), deterministic=True)
        def paired(reference, own):
            samples = {'reference': [], 'candidate': []}
            orders = []
            for repeat in range(4):
                order = [('reference', reference), ('candidate', own)]
                if repeat % 2: order.reverse()
                orders.append([label for label, _ in order])
                for label, fn in order:
                    bounded()
                    for _ in range(3): fn()
                    samples[label].append(clock.time_with_cuda_graph_us(fn))
            means = {k:statistics.mean(v[1:]) for k,v in samples.items()}
            return dict(samples_us=samples, mean_last3_us=means, orders=orders,
                sample_SD_us={k:statistics.stdev(v[1:]) for k,v in samples.items()},
                candidate_over_reference=means['candidate'] / means['reference'])
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            for role in a.roles:
                if role == 'ffn_gate_up':
                    gate = next(t for t in reader.tensors if t.name == 'blk.0.ffn_gate.weight')
                    up = next(t for t in reader.tensors if t.name == 'blk.0.ffn_up.weight')
                    tensor = fuse_along_output([portable(gate), portable(up)])
                    n, k = 39936, 6656
                else:
                    down = next(t for t in reader.tensors if t.name.endswith('.ffn_down.weight') and t.tensor_type == GGMLQuantizationType.Q4_K)
                    tensor = portable(down)
                    n, k = tuple(map(int, down.shape[::-1]))
                    assert (n, k) == (6656, 19968)
                module = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                pack_linear_for_cuda(module, {'weight': tensor})
                assert isinstance(module.weight, CudaCoalescedInt4Tensor)
                w = tuple(getattr(module.weight, name).to('cuda').contiguous()
                          for name in ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                report['weight_sources'][role] = dict(packed_SHA256=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in w],
                    N=n, K=k, current_ET_metadata_reencoding_unchanged=True)
                subset = tuple(t[:128].contiguous() for t in w)
                for arithmetic in ['bulk-staged-bf16', 'native-zero-fma-bf16']:
                    dense = (native_fma_dense if arithmetic == 'native-zero-fma-bf16' else staged_dense)(subset, 0, 128)
                    for bk in [32, 64, 128]:
                        actual = candidate.materialize(subset, arithmetic=arithmetic, bk=bk)
                        assert torch.equal(actual, dense), 'Effective packed-weight decoder mismatch'
                        report['materialization_checks'].append(dict(role=role, arithmetic=arithmetic, BK=bk,
                            weight_elements_observed=128*k, different_elements=0))
                        del actual
                    del dense
                del subset
                # Weight dimensions remain static; only activation M is dynamic.
                example = torch.randn((72, k), device='cuda', dtype=torch.bfloat16,
                                       generator=torch.Generator(device='cuda').manual_seed(32000+n))
                torch._dynamo.mark_dynamic(example, 0, min=5, max=2048)
                bulk = torch.compile(_dequant_matmul, fullgraph=True, dynamic=False, options=options)
                save('compiling-isolated-bulk-baseline', role=role)
                bulk(example, *w, 32)
                torch.cuda.synchronize()
                for entry in a.entries:
                    bounded()
                    kind = 'native' if entry.startswith('native') else 'bulk'
                    assert kind in ['native', 'bulk'] and entry.startswith(kind)
                    m = int(entry[len(kind):])
                    assert m in [8, 16, 64, 72, 128, 512, 2048]
                    arithmetic = 'native-zero-fma-bf16' if kind == 'native' else 'bulk-staged-bf16'
                    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16,
                                    generator=torch.Generator(device='cuda').manual_seed(33000+n+m))
                    x[0].zero_()
                    target = oracle(x, w, arithmetic)
                    metadata = {}
                    with set_wrap_triton_enabled(False):
                        if kind == 'native':
                            assert m in [8, 16, 64]
                            with autotune_launch_params() as selection:
                                native._launch_w4a16(m, x, *w, 32)
                            assert len(selection.picks) == 1
                            split = int(selection.picks[0].choice)
                            native._launch_w4a16(m, x, *w, 32, split_k=split)
                            best = (native._W4A16_DIRECT_KERNELS if split == 1 else native._W4A16_SPLIT_KERNELS)[m].best_config
                            config = triton.Config(dict(best.kwargs), num_warps=best.num_warps,
                                                   num_stages=best.num_stages, num_ctas=best.num_ctas)
                            reference = lambda: native._launch_w4a16(m, x, *w, 32, split_k=split, config=config)
                            metadata = dict(split=split, kwargs=dict(best.kwargs),
                                            warps=best.num_warps, stages=best.num_stages,
                                            eager_selected_not_current_AOTI_config=True)
                        else:
                            reference = lambda: bulk(x, *w, 32)
                        result = dict(role=role, entry=entry, M=m, N=n, K=k, arithmetic=arithmetic,
                                      reference_metadata=metadata, candidates=[])
                        report['results'].append(result)
                        result['reference_numeric'] = checked(reference, target)
                        for cfg in candidate.configs(m):
                            bounded()
                            record = dict(config=asdict(cfg), accepted=False)
                            result['candidates'].append(record)
                            save('checking-candidate', role=role, entry=entry, config=asdict(cfg))
                            own = out = partial = None
                            try:
                                out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
                                partial = (torch.empty((cfg.split, m, n), device=x.device, dtype=torch.float32)
                                           if cfg.split > 1 else None)
                                own = lambda: candidate.launch(x, w, config=cfg, arithmetic=arithmetic, out=out, partial=partial)
                                record['numeric'] = checked(own, target)
                                record['timing'] = paired(reference, own)
                                record['workspace_bytes'] = 0 if partial is None else partial.numel() * partial.element_size()
                                record['accepted'] = True
                            except Exception as error:
                                record.update(error_type=type(error).__name__, error=str(error))
                                if any(text in str(error).lower() for text in ['illegal memory access', 'device-side assert']):
                                    raise
                            finally:
                                del own, out, partial
                            save('candidate-screened', role=role, entry=entry)
                        valid = [r for r in result['candidates'] if r['accepted']]
                        if valid:
                            result['best'] = min(valid, key=lambda r:r['timing']['candidate_over_reference'])
                    del x, target, reference
                    torch.cuda.empty_cache()
                del w, module, tensor, example, bulk
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Importing a standalone candidate must not rewrite the ET decoder.
        assert digest(native.__file__) == report['native_source_SHA256']
        save('completed', cases=len(report['results']), model_route_changed=False)
    except Exception as error:
        (a.output / 'exception-traceback.txt').write_text(traceback.format_exc())
        save('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        fcntl.flock(held_lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    main()
