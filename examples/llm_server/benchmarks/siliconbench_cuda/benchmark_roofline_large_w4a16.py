"""Compare larger W4A16 buckets with the complete dequant/linear fallback.

Actual unchanged MG GGUF storage, synthetic BF16 inputs, FP32 accumulation.
Every configured split tunes the full tile space, with the existing K-loop
pruning. The selected launch is checked against native weight arithmetic;
fallback rounding differences are separately reported. Four AB/BA rounds,
discard first. This operator experiment cannot select a full-model crossover.
"""
import argparse
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['et-source', 'gguf', 'output', 'lock']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--roles', nargs='+', choices=['ffn_down', 'ffn_gate_up', 'attention_out'],
                   default=['ffn_down', 'ffn_gate_up', 'attention_out'])
    p.add_argument('--M', nargs='+', type=int, default=[64, 72, 128, 256, 512, 520])
    p.add_argument('--max-seconds', type=int, default=21600)
    p.add_argument('--actual-row-grid', action='store_true')
    p.add_argument('--stop-on-regression', action='store_true',
                   help='Stop each role at its first measured native slowdown')
    a = p.parse_args()
    assert all(5 <= m <= 16384 for m in a.M)
    if a.stop_on_regression:assert a.M == sorted(set(a.M))
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(cases=[], rounds=4, discard_first=True, activation='BF16', accumulation='FP32',
                  weights_requantized=False, new_KA=False, model_or_HTTP_accepted=False)

    def state(status, **fields):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')

    state('waiting-GPU-lock')
    try:
        with a.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor'),
                TRITON_CACHE_DIR=str(a.output / 'triton'), TORCHINDUCTOR_COMPILE_THREADS='1',
                MAX_JOBS='1', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            sys.path.insert(0, str(a.et_source.parent))
            import numpy as np
            import torch
            import triton
            from torch._library.triton import set_wrap_triton_enabled
            from gguf import GGUFReader, GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default, fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
            from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
            from public_grouped_w4a16 import materialize
            from roofline_large_w4a16 import install, configs
            assert torch.cuda.get_device_capability() == (12, 0)
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            assert sha(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            report.update(GGUF_SHA256=sha(a.gguf), kernel_extension=install(),
                native_SHA256=sha(native.__file__), extension_SHA256=sha(Path(__file__).with_name('roofline_large_w4a16.py')),
                packer_SHA256=sha(inspect.getfile(pack_linear_for_cuda)),
                device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=power.limit,clocks.max.sm,memory.total',
                                                    '--format=csv,noheader'], text=True).strip())
            report['actual_row_grid'] = a.actual_row_grid
            if a.actual_row_grid:
                from roofline_large_row_grid import install as install_row_grid
                install_row_grid()
            started = time.monotonic()

            def bounded():
                assert time.monotonic() - started < a.max_seconds, 'Preserve partial tuning; no automatic promotion'

            def portable(raw):
                assert raw.tensor_type == GGMLQuantizationType.Q4_K
                n = int(raw.shape[-1])
                data = torch.from_numpy(np.array(raw.data, copy=True)).view(torch.uint8).reshape(n, -1)
                return to_default(raw.name, ExportableGGUFTensor.from_raw(data, 'q4_k', torch.bfloat16))

            def numeric(y, target):
                d = y.float() - target.float()
                return dict(relative_RMS=float(d.square().mean().sqrt() /
                    target.float().square().mean().sqrt().clamp_min(1e-8)),
                    max_abs=float(d.abs().max()), finite=bool(torch.isfinite(y).all()))

            reader = GGUFReader(str(a.gguf))
            options = dict(max_autotune_gemm=True, max_autotune_gemm_backends='TRITON',
                           emulate_precision_casts=True, **{'triton.cudagraphs': False})
            report['fallback_compile_options'] = options
            tuning_events = []
            previous_listener = triton.knobs.autotuning.listener
            def record_pick(**event):
                tuning_events.append(event)
                if previous_listener is not None:
                    previous_listener(**event)
            triton.knobs.autotuning.listener = record_pick
            def launch_native(*args, **kwargs):
                # Match ET's launch-parameter tuner: eager wrap_triton would
                # clone the autotuner on each call and lose its selected cache.
                with set_wrap_triton_enabled(False):
                    return native._launch_w4a16(*args, **kwargs)
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream), clock.cuda_graph_timing() as timing_stats:
                for role in a.roles:
                    bounded()
                    names = ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight'] if role == 'ffn_gate_up' else []
                    if names:
                        raw_weights = [next(t for t in reader.tensors if t.name == name) for name in names]
                        weight = fuse_along_output([portable(t) for t in raw_weights])
                        n, k = 39936, 6656
                    else:
                        suffix = '.ffn_down.weight' if role == 'ffn_down' else '.attn_output.weight'
                        raw = next(t for t in reader.tensors if t.name.endswith(suffix)
                                   and t.tensor_type == GGMLQuantizationType.Q4_K)
                        raw_weights = [raw]
                        n, k = map(int, raw.shape[::-1]); weight = portable(raw)
                    module = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                    pack_linear_for_cuda(module, {'weight': weight})
                    w = tuple(getattr(module.weight, key).cuda().contiguous() for key in
                              ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                    oracle_weight = materialize(w, arithmetic='native-zero-fma-bf16', bk=128)
                    packed_hashes = [sha_bytes(v.cpu().numpy().tobytes()) for v in w]
                    for m in a.M:
                        bounded(); bucket = 1 << (m - 1).bit_length()
                        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16,
                            generator=torch.Generator(device='cuda').manual_seed(81000 + n + m))
                        reference = torch.compile(lambda x: _dequant_matmul(x, *w, 32), options=options)
                        state('compiling-complete-fallback', role=role, M=m, bucket=bucket)
                        reference(x); torch.cuda.synchronize()
                        native_target = torch.nn.functional.linear(x, oracle_weight)
                        candidates = []
                        for split in native.TILE_SPLIT_K_CANDIDATES:
                            bounded(); state('autotuning-native-large-bucket', role=role, M=m,
                                             bucket=bucket, split_k=split)
                            try:
                                tuning_events.clear()
                                y = launch_native(bucket, x, *w, 32, split_k=split)
                                torch.cuda.synchronize()
                            except (triton.runtime.errors.OutOfResources, InvalidLaunchParam) as error:
                                candidates.append(dict(split_k=split, unsupported=str(error))); continue
                            tuner = (native._W4A16_DIRECT_KERNELS if split == 1 else native._W4A16_SPLIT_KERNELS)[bucket]
                            picks = [e for e in tuning_events if tuple(e['key'][:4]) == (m,n,k,split)]
                            assert picks, 'Capture the actual autotune pick; do not substitute a default'
                            pick = picks[-1]
                            config = pick['best_config']
                            fn = lambda split=split, config=config: launch_native(
                                bucket, x, *w, 32, split_k=split, config=config)
                            ys = [fn().clone() for _ in range(3)]
                            torch.cuda.synchronize()
                            stats = numeric(ys[0], native_target)
                            assert stats['finite'] and stats['relative_RMS'] < .005
                            assert all(torch.equal(ys[0], value) for value in ys[1:])
                            duration = clock.time_with_cuda_graph_us(fn)
                            candidates.append(dict(split_k=split, us=duration, numeric=stats,
                                config=dict(kwargs=config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages),
                                declared_autotune_configs=len(configs(bucket)) if bucket >= 128 else len(native.int4_autotune_configs(bucket)),
                                config_timings=[dict(config=str(c), us=[float(v) for v in times] if isinstance(times,(list,tuple)) else float(times))
                                                for c,times in pick['configs_timings'].items()],
                                actual_autotune_key=[str(v) for v in pick['key']],
                                autotune_pick_from_runtime_listener=True))
                        valid = [v for v in candidates if 'us' in v]
                        assert valid, 'No usable native W4A16 launch'
                        winner = min(valid, key=lambda v: v['us'])
                        c = winner['config']
                        config = triton.Config(c['kwargs'], num_warps=c['num_warps'], num_stages=c['num_stages'])
                        candidate = lambda: launch_native(bucket, x, *w, 32,
                            split_k=winner['split_k'], config=config)
                        functions = dict(fallback=lambda: reference(x), candidate=candidate)
                        samples = {name: [] for name in functions}
                        for rnd in range(4):
                            order = ['fallback', 'candidate'] if rnd % 2 == 0 else ['candidate', 'fallback']
                            for name in order:
                                bounded()
                                for _ in range(3): functions[name]()
                                samples[name].append(clock.time_with_cuda_graph_us(functions[name]))
                        means = {name: statistics.mean(values[1:]) for name,values in samples.items()}
                        report['cases'].append(dict(role=role, M=m, N=n, K=k, bucket=bucket,
                            source_tensors=[v.name for v in raw_weights], packed_SHA256=packed_hashes,
                            tuning=candidates, winner=winner, samples_us=samples, mean_last3_us=means,
                            sample_SD_us={name: statistics.stdev(v[1:]) for name,v in samples.items()},
                            candidate_over_complete_fallback=means['candidate']/means['fallback'],
                            candidate_vs_fallback=numeric(candidate(), reference(x)),
                            fallback_vs_native_oracle=numeric(reference(x), native_target),
                            numerical_arithmetic_difference_disclosed=True))
                        report['autotune_graph_timing'] = dict(vars(timing_stats))
                        state('case-completed', role=role, M=m, completed_cases=len(report['cases']))
                        if a.stop_on_regression and means['candidate'] > means['fallback']:
                            report.setdefault('first_measured_regression', {})[role] = {
                                'M': m, 'candidate_over_complete_fallback': means['candidate']/means['fallback'],
                                'remaining_larger_M_skipped': [value for value in a.M if value > m]}
                            state('role-stopped-at-measured-regression', role=role, M=m)
                            break
                    del oracle_weight, w, module, weight
                    torch.cuda.empty_cache()
            state('completed-operator-screen-pending-full-forward-validation', completed_cases=len(report['cases']))
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc()); state('failed'); raise


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__': main()
