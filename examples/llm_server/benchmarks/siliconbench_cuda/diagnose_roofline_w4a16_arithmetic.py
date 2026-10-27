"""Diagnose large-M arithmetic with independent PyTorch weight reconstruction.

The original GGUF, packed codes, activation dtype and measured native launch
are preserved. Temporary dense weights are numerical controls only. This tool
does not install a serving route, change a PTE, or accept full-model numerics.
"""
import argparse
import datetime
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('et-source', 'gguf', 'policies', 'output', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--M', type=int, nargs='+', default=[128, 256, 512])
    args = parser.parse_args()
    assert args.M and all(65 <= m <= 1024 for m in args.M)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(roles=[], rounds=4, discarded_rounds=[1], averaged_rounds=[2, 3, 4],
                  full_model_accepted=False, main_results_changed=False,
                  dense_weights_are_diagnostic_only=True)

    def state(status, **extra):
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        (args.output / 'progress.json').write_text(json.dumps(dict(
            status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n')

    state('waiting-GPU-lock')
    try:
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            deadline = time.monotonic() + 120
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip():
                assert time.monotonic() < deadline, 'Existing CUDA context remains occupied'
                time.sleep(1)
            os.environ.update(TRITON_CACHE_DIR=str(args.output / 'triton'),
                              OMP_NUM_THREADS='2', MAX_JOBS='1')
            sys.path.insert(0, str(args.et_source.parent))
            import numpy as np
            import torch
            import torch.nn.functional as F
            import triton
            from gguf import GGUFReader, GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default, fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
            from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
            from executorch.backends.cuda.autotune import cuda_graph_timing as timing
            from public_grouped_w4a16 import materialize
            from roofline_large_row_grid import install

            assert torch.cuda.get_device_capability() == (12, 0)
            assert digest(native.__file__) == 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
            assert digest(args.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            install()
            policies = json.loads(args.policies.read_text())['policies']
            assert len(policies) == 3
            report.update(GGUF_SHA256=digest(args.gguf), policies_SHA256=digest(args.policies),
                          native_source_SHA256=digest(native.__file__), activation='BF16',
                          independent_oracle='PyTorch eager operations, FP64 zero product/subtraction before BF16 rounding',
                          GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,power.limit,driver_version',
                                                       '--format=csv,noheader'], text=True).strip())
            reader = GGUFReader(str(args.gguf))

            def portable(raw):
                assert raw.tensor_type == GGMLQuantizationType.Q4_K
                n = int(raw.shape[-1])
                data = torch.from_numpy(np.array(raw.data, copy=True)).view(torch.uint8).reshape(n, -1)
                gguf_weight = ExportableGGUFTensor.from_raw(data, 'q4_k', torch.bfloat16)
                return gguf_weight, to_default(raw.name, gguf_weight)

            def numeric(value, target):
                delta = value.float() - target.float()
                return dict(finite=bool(torch.isfinite(value).all()), bitwise_equal=bool(torch.equal(value, target)),
                            relative_RMS=float(delta.square().mean().sqrt() /
                                               target.float().square().mean().sqrt().clamp_min(1e-8)),
                            max_abs=float(delta.abs().max()), differing_elements=int((value != target).sum()))

            def independent_weights(weights):
                q, sc, ss, z, zs = weights
                n, half_k = q.shape
                k = 2 * half_k
                staged = torch.empty((n, k), device=q.device, dtype=torch.bfloat16)
                fused = torch.empty_like(staged)
                # Bound temporary FP64 working storage even for the largest role.
                for i in range(0, n, 128):
                    j = min(i + 128, n)
                    codes = torch.stack((q[i:j] & 15, q[i:j] >> 4), dim=-1).reshape(j-i, k).to(torch.bfloat16)
                    sg = (sc[i:j].float() * ss[i:j].to(torch.bfloat16).float().repeat_interleave(8, dim=1)).to(torch.bfloat16).repeat_interleave(32, dim=1)
                    zero_code = z[i:j].to(torch.bfloat16).repeat_interleave(32, dim=1)
                    zero_step = zs[i:j].to(torch.bfloat16).repeat_interleave(256, dim=1)
                    zg = (zero_code.float() * zero_step.float()).to(torch.bfloat16)
                    shifted_staged = (codes.float() - zg.float()).to(torch.bfloat16)
                    shifted_fused = (codes.double() - zero_code.double() * zero_step.double()).to(torch.bfloat16)
                    staged[i:j] = (shifted_staged.float() * sg.float()).to(torch.bfloat16)
                    fused[i:j] = (shifted_fused.float() * sg.float()).to(torch.bfloat16)
                return staged, fused

            with torch.inference_mode(), timing.cuda_graph_timing():
                for role, policy in zip(('ffn_down', 'ffn_gate_up', 'attention_out'), policies):
                    state('loading-original-role', role=role)
                    if role == 'ffn_gate_up':
                        raw = [next(t for t in reader.tensors if t.name == name)
                               for name in ('blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight')]
                    else:
                        suffix = '.ffn_down.weight' if role == 'ffn_down' else '.attn_output.weight'
                        raw = [next(t for t in reader.tensors if t.name.endswith(suffix)
                                    and t.tensor_type == GGMLQuantizationType.Q4_K)]
                    pairs = [portable(t) for t in raw]
                    weight = pairs[0][1] if len(pairs) == 1 else fuse_along_output([p[1] for p in pairs])
                    n, k = policy['shape']
                    module = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                    pack_linear_for_cuda(module, {'weight': weight})
                    weights = tuple(getattr(module.weight, name).cuda().contiguous()
                                    for name in ('qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'))
                    hashes = [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in weights]
                    assert hashes == policy['packed_SHA256']
                    state('independent-weight-reconstruction', role=role)
                    staged, fused = independent_weights(weights)
                    checks = {}
                    for name, arithmetic, target in (('staged', 'bulk-staged-bf16', staged),
                                                     ('fused', 'native-zero-fma-bf16', fused)):
                        dense = materialize(weights, arithmetic=arithmetic, bk=128)
                        checks[name + '_Triton_vs_independent_PyTorch'] = numeric(dense, target)
                        assert checks[name + '_Triton_vs_independent_PyTorch']['bitwise_equal']
                        del dense
                    checks['fused_vs_staged_weights'] = numeric(fused, staged)
                    original = torch.cat([pair[0].dequantize(torch.float32) for pair in pairs]).cuda()
                    checks['staged_vs_original_GGUF_FP32'] = numeric(staged, original)
                    checks['fused_vs_original_GGUF_FP32'] = numeric(fused, original)
                    original_bf16 = original.to(torch.bfloat16)
                    del original
                    item = dict(role=role, original_tensor_names=[t.name for t in raw],
                                shape=[n, k], packed_SHA256=hashes, weight_checks=checks, cases=[])
                    report['roles'].append(item)
                    for m in args.M:
                        branch = next((b for b in policy['branches'] if m <= b['max_M']), None)
                        state('actual-native-GEMM-control', role=role, M=m)
                        x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda',
                                        generator=torch.Generator(device='cuda').manual_seed(43000 + n + m))
                        if branch:
                            c = branch['config']
                            config = triton.Config(dict(c['kwargs']), num_warps=c['num_warps'], num_stages=c['num_stages'])
                            candidate = lambda: native._launch_w4a16(branch['bucket'], x, *weights, 32,
                                                                   split_k=branch['split_k'], config=config)
                        else:
                            candidate = lambda: _dequant_matmul(x, *weights, 32)
                        fallback = lambda: _dequant_matmul(x, *weights, 32)
                        outputs = dict(native=candidate().clone(), fallback=fallback().clone(),
                                       independent_staged=F.linear(x, staged), independent_fused=F.linear(x, fused),
                                       original_GGUF_bf16=F.linear(x, original_bf16))
                        target = outputs['independent_fused'] if branch else outputs['independent_staged']
                        checks = dict(native_vs_own_independent_oracle=numeric(outputs['native'], target),
                                      fallback_vs_independent_staged=numeric(outputs['fallback'], outputs['independent_staged']),
                                      native_vs_fallback=numeric(outputs['native'], outputs['fallback']),
                                      independent_fused_vs_staged=numeric(outputs['independent_fused'], outputs['independent_staged']),
                                      native_vs_original_GGUF_bf16=numeric(outputs['native'], outputs['original_GGUF_bf16']),
                                      fallback_vs_original_GGUF_bf16=numeric(outputs['fallback'], outputs['original_GGUF_bf16']))
                        assert checks['native_vs_own_independent_oracle']['finite']
                        assert checks['fallback_vs_independent_staged']['bitwise_equal']
                        samples = dict(native=[], fallback=[])
                        functions = dict(native=candidate, fallback=fallback)
                        for rnd in range(4):
                            for name in (('fallback', 'native') if rnd % 2 == 0 else ('native', 'fallback')):
                                for _ in range(3):
                                    functions[name]()
                                samples[name].append(timing.time_with_cuda_graph_us(functions[name]))
                        raw_path = args.output / (role + '-M' + str(m) + '-outputs.npz')
                        np.savez_compressed(raw_path, activation=x.float().cpu().numpy(),
                                            **{name: value.float().cpu().numpy() for name, value in outputs.items()})
                        item['cases'].append(dict(M=m, selected_branch=branch, checks=checks,
                                                  samples_us=samples, mean_last3_us={name: statistics.mean(v[1:]) for name,v in samples.items()},
                                                  sample_SD_us={name: statistics.stdev(v[1:]) for name,v in samples.items()},
                                                  raw_outputs=dict(path=str(raw_path), SHA256=digest(raw_path), bytes=raw_path.stat().st_size)))
                        del x, outputs
                        state('case-completed', role=role, M=m)
                    del weights, staged, fused, original_bf16, module, weight, pairs
                    torch.cuda.empty_cache()
                state('completed-arithmetic-diagnostic-full-model-acceptance-pending')
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        state('failed')
        raise


if __name__ == '__main__':
    main()
