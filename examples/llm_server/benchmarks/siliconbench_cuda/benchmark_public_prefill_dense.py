"""Bounded SM120 dense/persistent and full-dequant pipeline prefill screen.

Reads original public GGUF and ET packing locally. Activation stays BF16,
weights retain original quantized storage and staged BF16 decode rounding.
Dense-only timings are diagnostic; full-pipeline timings include dequant.
Four paired AB/BA rounds, discard first. No model changes or automatic winner.
"""
import argparse
from dataclasses import asdict
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8 * 1024 * 1024): h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--after-runtime-job', type=Path)
    p.add_argument('--max-seconds', type=int, default=5400)
    p.add_argument('--M', nargs='+', type=int, default=[512, 520, 2048])
    p.add_argument('--resource-followup', action='store_true')
    a = p.parse_args()
    assert set(a.M) <= {512, 520, 2048} and not a.output.exists()
    a.output.mkdir(parents=True)
    report = dict(controller_pid=os.getpid(), scope=__doc__, results=[],
                  model_dispatch_changed=False, new_KA=False,
                  activation_dtype='BF16', accumulation='FP32',
                  weights_requantized=False, full_model_or_HTTP_accepted=False,
                  rounds=4, discard_first=True, resource_followup=a.resource_followup,
                  reference='Unmodified ET staged dequant + Inductor F.linear')
    def state(status, **fields):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')
    try:
        if a.after_runtime_job:
            state('waiting-existing-mini-runtime')
            deadline = time.monotonic() + 20000
            while True:
                prior = json.loads((a.after_runtime_job / 'progress.json').read_text())
                if prior['status'].startswith('completed') or prior['status'] == 'failed':
                    # Ensure the preceding parent has released its GPU lock.
                    try: os.kill(int(prior['controller_pid']), 0)
                    except ProcessLookupError: break
                else: os.kill(int(prior['controller_pid']), 0)
                assert time.monotonic() < deadline, 'Observation expired; do not restart preceding jobs'
                time.sleep(20)
        state('waiting-GPU-lock')
        with a.lock.open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            started = time.monotonic()
            def bounded():
                assert time.monotonic() - started < a.max_seconds, 'Dense screening time budget expired'
                owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).splitlines()
                assert all(int(v) == os.getpid() for v in owners if v.strip()), 'Other GPU owner present'
            bounded()
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor-cache'),
                TRITON_CACHE_DIR=str(a.output / 'triton-cache'),
                TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', OMP_NUM_THREADS='2')
            sys.path.insert(0, str(a.et_source.parent))
            import numpy as np
            import torch
            import triton
            from gguf import GGUFReader, GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default, fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            import public_grouped_w4a16 as packed
            import public_prefill_dense as candidate
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            assert torch.cuda.get_device_capability() == (12, 0)
            assert sha(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            assert sha(inspect.getfile(_dequant_matmul)) == '1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'
            assert sha(inspect.getfile(pack_linear_for_cuda)) == 'c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
            assert sha(inspect.getfile(to_default)) == '5daca6d9330b9b32a75dbcc8076a8c17dc433402509a87dff06a27ea1f22eeb5'
            report.update(GGUF_SHA256=sha(a.gguf), capability=[12, 0], device=torch.cuda.get_device_name(),
                torch=torch.__version__, Triton=triton.__version__, candidate_SHA256=sha(candidate.__file__),
                packed_decoder_SHA256=sha(packed.__file__), native_dispatch_SHA256=sha(inspect.getfile(_dequant_matmul)))
            options = dict(max_autotune_gemm=True, max_autotune_gemm_backends='TRITON',
                           emulate_precision_casts=True, **{'triton.cudagraphs': False})
            report['compile_options'] = options
            reader = GGUFReader(str(a.gguf))
            def portable(t):
                assert t.tensor_type == GGMLQuantizationType.Q4_K
                n, _ = map(int, t.shape[::-1])
                raw = torch.from_numpy(np.array(t.data, copy=True)).view(torch.uint8).reshape(n, -1)
                return to_default(t.name, ExportableGGUFTensor.from_raw(raw, 'q4_k', torch.bfloat16))
            def checked(fn, target):
                ys = [fn().clone() for _ in range(4)]
                torch.cuda.synchronize()
                assert all(torch.equal(ys[1], y) for y in ys[2:])
                y = ys[1]; assert y.dtype == target.dtype == torch.bfloat16 and y.shape == target.shape
                assert bool(torch.isfinite(y).all())
                delta = y.float() - target.float()
                rms = float(delta.square().mean().sqrt() / target.float().square().mean().sqrt().clamp_min(1e-8))
                assert rms < .005, rms
                torch.testing.assert_close(y.float(), target.float(), rtol=.05, atol=4.0)
                return dict(relative_RMS=rms, max_abs=float(delta.abs().max()), deterministic=True)
            def paired(reference, own):
                samples = dict(reference=[], candidate=[])
                orders = []
                for r in range(4):
                    order = [('reference', reference), ('candidate', own)]
                    if r % 2: order.reverse()
                    orders.append([k for k, _ in order])
                    for name, fn in order:
                        bounded()
                        for _ in range(3): fn()
                        samples[name].append(clock.time_with_cuda_graph_us(fn))
                mean = {k: statistics.mean(v[1:]) for k, v in samples.items()}
                return dict(samples_us=samples, mean_last3_us=mean, orders=orders,
                            sample_SD_us={k: statistics.stdev(v[1:]) for k, v in samples.items()},
                            candidate_over_reference=mean['candidate'] / mean['reference'])
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream):
                for role in ['ffn_gate_up', 'ffn_down']:
                    if role == 'ffn_gate_up':
                        t = fuse_along_output([portable(next(t for t in reader.tensors if t.name == 'blk.0.ffn_'+name+'.weight')) for name in ['gate', 'up']])
                        n, k = 39936, 6656
                    else:
                        raw = next(t for t in reader.tensors if t.name.endswith('.ffn_down.weight') and t.tensor_type == GGMLQuantizationType.Q4_K)
                        t = portable(raw); n, k = map(int, raw.shape[::-1]); assert (n, k) == (6656, 19968)
                    module = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                    pack_linear_for_cuda(module, {'weight': t})
                    w = tuple(getattr(module.weight, name).to('cuda').contiguous() for name in ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                    dense = packed.materialize(w, arithmetic='bulk-staged-bf16', bk=128)
                    # Compare the entire materialized weight to independent staged
                    # BF16 tensor arithmetic, one row block at a time.
                    for start in range(0, n, 128):
                        end = min(start + 128, n)
                        q, sc, ss, z, zs = [v[start:end] for v in w]
                        codes = torch.stack((q & 15, q >> 4), -1).bfloat16().reshape(end-start, k//32, 32)
                        sg = (sc.bfloat16() * ss.bfloat16().repeat_interleave(8, 1)).bfloat16()
                        zg = (z.bfloat16() * zs.bfloat16().repeat_interleave(8, 1)).bfloat16()
                        expected = ((codes - zg[:, :, None]).bfloat16() * sg[:, :, None]).bfloat16().reshape(end-start, k)
                        assert torch.equal(dense[start:end], expected), 'Staged BF16 weight decode changed'
                    weight_info = dict(role=role, shape=[n,k], packed_SHA256=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in w],
                                       decoded_weight_elements=n*k, different_decoded_elements=0)
                    del q, sc, ss, z, zs, codes, sg, zg, expected
                    example = torch.randn((2048, k), device='cuda', dtype=torch.bfloat16)
                    torch._dynamo.mark_dynamic(example, 0, min=128, max=2048)
                    dense_reference = torch.compile(torch.nn.functional.linear, fullgraph=True, dynamic=False, options=options)
                    full_reference = torch.compile(_dequant_matmul, fullgraph=True, dynamic=False, options=options)
                    state('compiling-dense-and-full-reference', role=role)
                    dense_reference(example, dense); full_reference(example, *w, 32); torch.cuda.synchronize()
                    del example
                    for m in a.M:
                        x = torch.randn((m,k), device='cuda', dtype=torch.bfloat16, generator=torch.Generator(device='cuda').manual_seed(47000+n+m))
                        x[0].zero_(); out = torch.empty((m,n), device='cuda', dtype=torch.bfloat16)
                        target = torch.nn.functional.linear(x.float(), dense.float()).bfloat16()
                        reference = lambda: dense_reference(x, dense)
                        full = lambda: full_reference(x, *w, 32)
                        ref_numeric = checked(reference, target); full_numeric = checked(full, target)
                        cases = []
                        for config in candidate.configs(resource_followup=a.resource_followup):
                            bounded(); state('screening-dense', role=role, M=m, config=asdict(config))
                            fn = lambda config=config: candidate.launch(x, dense, out, config)
                            try:
                                numeric = checked(fn, target)
                                timing = paired(reference, fn)
                                cases.append(dict(config=asdict(config), accepted_numeric=True, numeric=numeric, timing=timing))
                            except Exception as error:
                                cases.append(dict(config=asdict(config), accepted_numeric=False, error=str(error)))
                        winners = [v for v in cases if v['accepted_numeric']]
                        assert winners
                        best = min(winners, key=lambda v: v['timing']['mean_last3_us']['candidate'])
                        selected = candidate.Config(**best['config'])
                        def pipeline():
                            packed._materialize[(triton.cdiv(n,32),k//128)](*w, dense, N=n, K=k, BN=32, BK=128,
                                ZERO_FMA=False, num_warps=4, num_stages=1, enable_fp_fusion=False)
                            return candidate.launch(x, dense, out, selected)
                        pipeline_numeric = checked(pipeline, target)
                        pipeline_timing = paired(full, pipeline)
                        # Native cuBLAS is diagnostic only; ET's model remains Triton.
                        cublas = lambda: torch.nn.functional.linear(x, dense)
                        cublas_numeric = checked(cublas, target)
                        cublas_timing = paired(reference, cublas)
                        report['results'].append(dict(role=role, M=m, N=n, K=k, weight=weight_info,
                            dense_reference_numeric=ref_numeric, full_reference_numeric=full_numeric,
                            configs=cases, best_dense=best,
                            selected_full_pipeline=dict(numeric=pipeline_numeric, timing=pipeline_timing),
                            cublas_dense=dict(numeric=cublas_numeric, timing=cublas_timing),
                            dense_only_excludes_unpack=True, selected_pipeline_includes_unpack=True))
                        state('case-completed', role=role, M=m)
                        del x, out, target
                    del dense, w, t, module, dense_reference, full_reference
                    torch.cuda.empty_cache()
            state('completed-standalone-dense-screen-pending-model-validation')
    except BaseException as error:
        report['error'] = str(error); report['traceback'] = traceback.format_exc()
        state('failed'); raise


if __name__ == '__main__': main()
