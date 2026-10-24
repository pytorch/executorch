"""Diagnose native GEMM versus staged oracle on one actual MG GGUF gate matrix.

Synthetic BF16 activations; exact source GGUF, portable fusion and native CUDA
packer. Native staged-BF16 and candidate FP32-affine operands have separate
numerical controls. Kernel timings do not certify whole-server performance or
quality. No runtime, dispatch policy, model or serving bucket is modified.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['et-source', 'gguf', 'output', 'lock']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--owned-worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    cutoff = datetime.datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
    if not a.owned_worker:
        assert not a.output.exists(), 'Preserve prior attempts'
        a.output.mkdir(parents=True)
        (a.output / 'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid())) + '\n')
        (a.output / 'progress.json').write_text(json.dumps(dict(status='waiting-GPU-lock', controller_pid=os.getpid())) + '\n')
        with a.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            remaining = (cutoff - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            assert remaining > 0, 'User tuning cutoff passed'
            with (a.output / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--owned-worker'],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    assert child.wait(timeout=min(7200, remaining)) == 0, 'Inspect preserved worker results'
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    (a.output / 'progress.json').write_text(json.dumps(dict(status='failed', reason='User cutoff or two-hour bound')) + '\n')
                    raise
            until = time.monotonic() + 60
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                assert time.monotonic() < until, 'Context teardown incomplete; retain GPU lock'
                time.sleep(1)
        return
    assert json.loads((a.output / 'lock-owner.json').read_text())['pid'] == os.getppid()
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
                      TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', TORCH_CUDA_ARCH_LIST='12.0',
                      OMP_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1',
                      TRITON_CACHE_DIR=str(a.output / 'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor-cache'))
    sys.path.insert(0, str(a.et_source.parent))
    import numpy as np
    import inspect
    import torch
    import triton
    from gguf import GGUFReader, GGMLQuantizationType
    from torch._library.triton import set_wrap_triton_enabled
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from executorch.extension.llm.export.quant import to_default, fuse_along_output
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    from executorch.backends.cuda.triton.kernels import quantized_gemm_family as family
    from executorch.backends.cuda.autotune import cuda_graph_timing as clock
    from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
    from candidates_w4a16_v8 import CONFIGS, run
    from candidates_w4a16_v8.aoti import candidate_op, config_index
    from candidates_w4a16_v8.validate import precision_oracle, staged_oracle, error_stats
    TIMING_POLICY='unchanged native cuda_graph_timing.time_with_cuda_graph_us; cold L2 per call; 4-32 calls in one graph; minimum GPU event interval'
    from aoti_single_threaded import SingleThreadedPackage, INVOCATION_POLICY
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.get_device_capability() == (12, 0)
    assert digest_file(Path(native.__file__)) == 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    assert digest_file(Path(inspect.getfile(pack_linear_for_cuda))) == 'c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
    assert digest_file(Path(inspect.getfile(to_default))) == '5daca6d9330b9b32a75dbcc8076a8c17dc433402509a87dff06a27ea1f22eeb5'
    assert digest_file(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'public-source-manifest.json').read_text())
    for name, proof in manifest['files'].items():
        assert digest_file(here / name) == proof['sha256']
    report = dict(scope=__doc__, GGUF_sha256=digest_file(a.gguf), public_source_release=str(here),
                  public_source_manifest_sha256=digest_file(here / 'public-source-manifest.json'),
                  native_source_sha256=digest_file(Path(native.__file__)),
                  model_packer_source_sha256=digest_file(Path(inspect.getfile(pack_linear_for_cuda))),
                  portable_fusion_source_sha256=digest_file(Path(inspect.getfile(fuse_along_output))),
                  device=torch.cuda.get_device_name(), capability=[12, 0], torch=torch.__version__, triton=triton.__version__,
                  rounds=4, discarded=[1], averaged=[2, 3, 4], orders=['AB', 'BA', 'AB', 'BA'],
                  native_arithmetic='Existing staged BF16 affine reconstruction',
                  candidate_arithmetic='FP32 affine reconstruction, final BF16 MMA operand',
                  same_arithmetic_claim=False, activation_source='Synthetic BF16',
                  weight_source='Actual unchanged MG GGUF, native portable conversion/fusion/CUDA packing',
                  default_pending_MG_policy_unchanged=True, automatic_deployment=False,
                  server_performance_F1_memory_acceptance_pending=True, A100_acceptance_separate=True,
                  cases=[], AOTI_cases=[])

    def save(status, **fields):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')

    def bounded():
        assert datetime.datetime.now(datetime.timezone.utc) < cutoff, 'User tuning cutoff reached'
        ids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).splitlines()
        assert all(int(pid) == os.getpid() for pid in ids if pid.strip()), 'Other GPU owner detected'

    def checked(fn, oracle):
        ys = [fn() for _ in range(4)]
        torch.cuda.synchronize()
        assert all(torch.equal(ys[1], y) for y in ys[2:]), 'Nondeterministic output'
        return dict(deterministic=True, **error_stats(ys[1], oracle))

    def difference(y, oracle):
        d = y.float() - oracle.float()
        return dict(relative_rms=float(d.square().mean().sqrt() / oracle.float().square().mean().sqrt().clamp_min(1e-8)),
                    max_abs=float(d.abs().max()), diagnostic_only=True)

    def paired(reference, candidate, timer=None):
        timer = clock.time_with_cuda_graph_us if timer is None else timer
        samples = {'reference': [], 'candidate': []}
        orders=[]
        for repeat in range(4):
            order = [('reference', reference), ('candidate', candidate)]
            if repeat % 2:
                order.reverse()
            orders.append([label for label,_ in order])
            for label, fn in order:
                bounded()
                for _ in range(3):
                    fn()
                samples[label].append(timer(fn))
        means = {label: statistics.mean(v[1:]) for label, v in samples.items()}
        return dict(samples_us=samples, means_us=means, orders=orders,
                    sample_sd_us={label: statistics.stdev(v[1:]) for label, v in samples.items()},
                    candidate_over_reference=means['candidate'] / means['reference'])

    def audit_aoti(role, w, baseline, cfg):
        bounded()
        bucket = 8
        n, k = int(w[0].shape[0]), int(w[0].shape[1]) * 2
        x = torch.randn((bucket, k), device='cuda', dtype=torch.bfloat16,
                        generator=torch.Generator(device='cuda').manual_seed(27000 + n))
        ident = 'v8_m8_mg_sp1_pending'
        assert CONFIGS[ident]['bucket'] == bucket
        def reference_launch(row_bucket, x, q, sc, ss, z, zs, group_size):
            assert row_bucket == bucket and group_size == 32
            return native._launch_w4a16(bucket, x, q, sc, ss, z, zs, 32, split_k=baseline['split_k'], config=cfg)
        registered = family.QuantizedGemmFamily('ka_v8_actual_mg_' + role, [bucket],
                                               native._prototype, reference_launch, native._fake, native._unsupported_reason)
        op = registered.op(bucket)
        class Reference(torch.nn.Module):
            def forward(self, x, q, sc, ss, z, zs):
                return op(x, q, sc, ss, z, zs, 32)
        class Candidate(torch.nn.Module):
            def forward(self, x, q, sc, ss, z, zs):
                return candidate_op(x, q, sc, ss, z, zs, config_index(ident))
        record = dict(role=role, bucket=bucket, n=n, k=k, config_id=ident,
                      dynamic_M_min=5, dynamic_M_max=8, pending_policy_not_promoted=True,
                      reference_metadata=baseline, packages={}, tail_numerics=[])
        report['AOTI_cases'].append(record)
        models = {}
        directory = a.output / 'aoti-packages'
        directory.mkdir(exist_ok=True)
        dynamic = ({0: torch.export.Dim('active_M', min=5, max=8)}, None, None, None, None, None)
        for label, module in [('reference', Reference()), ('candidate', Candidate())]:
            bounded()
            save('compiling-AOTI-package', role=role, label=label)
            ep = torch.export.export(module, (x, *w), dynamic_shapes=dynamic, strict=True)
            package = directory / (role + '-m8-' + label + '.pt2')
            torch._inductor.aoti_compile_and_package(ep, package_path=str(package))
            models[label] = SingleThreadedPackage(package,directory/'official-api-extracted'/package.stem)
            record['packages'][label] = dict(path=str(package), bytes=package.stat().st_size,
                                             sha256=digest_file(package), export_range_constraints=str(ep.range_constraints))
        def call(label, xx):
            y = models[label](xx, *w)
            if isinstance(y, (tuple, list)):
                assert len(y) == 1
                y = y[0]
            return y
        for m in [5, 7, 8]:
            bounded()
            xx = x[:m].contiguous()
            candidate_oracle, native_oracle = precision_oracle(xx, w), staged_oracle(xx, w)
            record['tail_numerics'].append(dict(m=m, checks={
                'reference': checked(lambda: call('reference', xx), native_oracle),
                'candidate': checked(lambda: call('candidate', xx), candidate_oracle)},
                candidate_versus_native=difference(call('candidate', xx), call('reference', xx))))
        record['paired_timing'] = paired(lambda: call('reference', x), lambda: call('candidate', x),
                                        timer=clock.time_with_cuda_graph_us)
        record['AOTI_timing_policy'] = TIMING_POLICY
        record['AOTI_invocation_policy'] = INVOCATION_POLICY
        record['extracted_package_members'] = {k:v.members for k,v in models.items()}
        save('AOTI-package-load-validated', role=role)
        for model in models.values():model.close()
        del models
        torch.cuda.empty_cache()

    try:
        reader = GGUFReader(str(a.gguf))
        tensors = {role: next(t for t in reader.tensors if t.name == 'blk.0.' + role + '.weight')
                   for role in ['ffn_gate', 'ffn_up']}
        portable = {}
        source_records = {}
        for role, tensor in tensors.items():
            assert tensor.tensor_type == GGMLQuantizationType.Q4_K
            assert tuple(map(int, tensor.shape[::-1])) == (19968, 6656)
            raw = torch.from_numpy(np.array(tensor.data, copy=True)).view(torch.uint8).reshape(19968, -1)
            portable[role] = to_default(tensor.name, ExportableGGUFTensor.from_raw(raw, 'q4_k', torch.bfloat16))
            source_records[role] = dict(name=tensor.name, quant='Q4_K', raw_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())
        portable['ffn_gate_up'] = fuse_along_output([portable['ffn_gate'], portable['ffn_up']])
        source_records['ffn_gate_up'] = dict(ordered_source_roles=['ffn_gate', 'ffn_up'], fusion_before_CUDA_packing=True)
        report['GGUF_tensors'] = source_records
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            for role in ['ffn_gate', 'ffn_up', 'ffn_gate_up']:
                expected_n = 39936 if role == 'ffn_gate_up' else 19968
                module = torch.nn.Linear(6656, expected_n, bias=False, device='meta', dtype=torch.bfloat16)
                pack_linear_for_cuda(module, {'weight': portable[role]})
                packed = module.weight
                assert isinstance(packed, CudaCoalescedInt4Tensor)
                w = tuple(getattr(packed, name).to('cuda').contiguous()
                          for name in ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                n, k = int(w[0].shape[0]), int(w[0].shape[1]) * 2
                assert k == 6656 and n == (39936 if role == 'ffn_gate_up' else 19968)
                weight_hashes = [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in w]
                m8_baseline = None
                for m in [8, 16]:
                    bounded()
                    save('running-native-preparation', role=role, m=m)
                    x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda',
                                    generator=torch.Generator(device='cuda').manual_seed(26000 + n + m))
                    candidate_oracle, native_oracle = precision_oracle(x, w), staged_oracle(x, w)
                    with set_wrap_triton_enabled(False):
                        with autotune_launch_params() as selection:
                            native._launch_w4a16(m, x, *w, 32)
                        assert len(selection.picks) == 1
                        pick = selection.picks[0]
                        assert pick.function.endswith('._launch_w4a16') and pick.param == 'split_k'
                        split = int(pick.choice)
                        native._launch_w4a16(m, x, *w, 32, split_k=split)
                        kernels = native._W4A16_DIRECT_KERNELS if split == 1 else native._W4A16_SPLIT_KERNELS
                        best = kernels[m].best_config
                        cfg = triton.Config(dict(best.kwargs), num_warps=best.num_warps, num_stages=best.num_stages, num_ctas=best.num_ctas)
                        reference = lambda: native._launch_w4a16(m, x, *w, 32, split_k=split, config=cfg)
                        metadata = dict(split_k=split, split_times_us=pick.times_us, eagerly_selected=True,
                                        executable_selected_config_unproven=True,
                                        config=dict(kwargs=cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages, num_ctas=cfg.num_ctas))
                        from native_gemm_oracle_diagnostic import diagnose
                        report['native_selected_config']=metadata
                        report['native_oracle_context']=dict(role=role,m=m,n=n,k=k,packed_weight_sha256=weight_hashes,
                            input_dtypes=[str(t.dtype) for t in w],all_GGUF_weights_and_native_source_unchanged=True)
                        report['native_oracle_diagnostic']=diagnose(native,x,w,reference,a.output,
                            reference_with_input=lambda xx:native._launch_w4a16(m,xx,*w,32,split_k=split,config=cfg))
                        report['no_candidate_numeric_or_timing_acceptance_claim']=True
                        save('completed-native-oracle-diagnostic',role=role,m=m)
                        return
                        record = dict(role=role, M=m, N=n, K=k, packed_weight_sha256=weight_hashes,
                                      reference_metadata=metadata, reference_numerics=checked(reference, native_oracle),
                                      native_versus_candidate_oracle=difference(reference(), candidate_oracle), candidates=[])
                        report['cases'].append(record)
                        for ident, candidate_cfg in CONFIGS.items():
                            if candidate_cfg['bucket'] != m:
                                continue
                            own = lambda ident=ident: run(x, *w, bucket=m, config=ident)
                            record['candidates'].append(dict(id=ident, config=candidate_cfg,
                                numerics=checked(own, candidate_oracle), candidate_versus_native=difference(own(), reference()),
                                paired_timing=paired(reference, own)))
                            save('running-candidate-pairs', role=role, m=m, candidate=ident)
                    if m == 8:
                        m8_baseline = (metadata, cfg)
                    del x, candidate_oracle, native_oracle, reference
                if role in ['ffn_gate', 'ffn_gate_up']:
                    audit_aoti(role, w, *m8_baseline)
                del w, packed, module
                torch.cuda.empty_cache()
        save('completed', cases=len(report['cases']), candidate_records=sum(len(c['candidates']) for c in report['cases']),
             AOTI_cases=len(report['AOTI_cases']), server_acceptance_pending=True)
    except Exception as error:
        (a.output / 'exception-traceback.txt').write_text(traceback.format_exc())
        save('failed', error_type=type(error).__name__, detail=str(error)[-1200:])
        raise


if __name__ == '__main__':
    main()
