"""Export solo/text-only MG with capacity for active sessions and prefix clones.

Keep the validated source, weights, static methods and BF16 activation policy.
The artifact is a candidate; successful export is not server/F1/performance
acceptance. A CPU-only parent owns the GPU lock through child process teardown.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--et-source', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--reference-deployment', type=Path, required=True)
    parser.add_argument('--max-kv-seqs', type=int, choices=[18,21,33], default=18)
    parser.add_argument('--max-step-tokens', type=int, choices=[2048,8192], default=8192)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--owned-gpu-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--max-context', type=int, choices=[12288,131072], default=12288)
    args = parser.parse_args()

    def state(status, **fields):
        pending = args.job_dir / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          max_kv_seqs=args.max_kv_seqs,
                                          max_step_tokens=args.max_step_tokens, **fields), indent=2) + '\n')
        pending.replace(args.job_dir / 'progress.json')

    if not args.owned_gpu_worker:
        assert not args.output.exists() and not args.job_dir.exists(), 'Preserve prior artifacts and attempts'
        args.job_dir.mkdir(parents=True)
        (args.job_dir / 'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid())) + '\n')
        state('waiting-GPU-lock')
        try:
            with args.lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                deadline = time.monotonic() + 120
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip():
                    assert time.monotonic() < deadline, 'GPU remains occupied; leave existing work untouched'
                    time.sleep(5)
                with (args.job_dir / 'export.log').open('x') as log:
                    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                    *sys.argv[1:], '--owned-gpu-worker'], stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=21600)
                assert json.loads((args.job_dir / 'progress.json').read_text())['status'] == 'artifact-exported-and-audited'
                deadline = time.monotonic() + 60
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip():
                    assert time.monotonic() < deadline, 'Wait for completed CUDA context teardown'
                    time.sleep(1)
                state('completed', artifact=str(args.output), deployment_accepted=False,
                      performance_memory_and_GMRID_acceptance_pending=True)
        except Exception as exc:
            state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
            raise
        return

    assert json.loads((args.job_dir / 'lock-owner.json').read_text())['pid'] == os.getppid()
    with args.lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Child requires its parent to own the shared GPU lock')
    assert not args.output.exists()
    assert digest(args.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    reference = json.loads(args.reference_deployment.read_text())
    sources = {}
    kernels = args.et_source / 'backends/cuda/triton/kernels'
    for name, expected in reference['quant_source_sha256'].items():
        assert name in ['int4_quantized_gemm.py', 'int6_quantized_gemm.py', 'quantized_gemm_utils.py']
        assert digest(kernels / name) == expected
        sources[name] = expected
    assert set(sources) == {'int4_quantized_gemm.py', 'int6_quantized_gemm.py', 'quantized_gemm_utils.py'}
    assert digest(kernels / 'sdpa_varlen.py') == '5ce65105639785e625739c90c0b208a77d12f8544208c85aa66e143d34a4e8ea'
    cache = args.job_dir / 'cache'
    (cache / 'host-zero-backed').mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
                      TORCH_CUDA_ARCH_LIST='12.0', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                      ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache / 'host-zero-backed'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                      OMP_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(args.et_source.parent))
    import torch
    import torch._inductor.cpp_builder as cpp
    assert torch.cuda.get_device_capability() == (12,0)
    original_build = cpp.CppBuilder.build

    def record_build(self):
        flags = re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)', self.get_command_line())
        with (args.job_dir / 'actual-compiler-optimization.jsonl').open('a') as log:
            log.write(json.dumps(dict(optimization_flags=flags, target_name=self._name)) + '\n')
        return original_build(self)

    cpp.CppBuilder.build = record_build
    state('exporting')
    sys.argv = ['export_solo', '--backend', 'cuda', '--gguf', str(args.gguf),
                '--use-offgraph-kv-cache', '--kv-layout', 'sequence',
                '--max-kv-seqs', str(args.max_kv_seqs), '--max-seq-len', str(args.max_context),
                '--forward-sizes', '1,2,4,8,16,32,64', '--max-step-tokens', str(args.max_step_tokens),
                '--activation-dtype', 'bfloat16', '--output-dir', str(args.output)]
    runpy.run_module('executorch.examples.models.muse_glimmer.export.export_solo', run_name='__main__')
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import Bool, Int, String
    from executorch.exir.scalar_type import ScalarType
    program = deserialize_pte_binary((args.output / 'model.pte').read_bytes()).program
    methods = {plan.name: plan for plan in program.execution_plan}
    forwards = {f'forward_{m}' for m in [1,2,4,8,16,32,64]} | {'forward_others'}
    assert forwards | {'sample', 'sample_argmax'} <= set(methods)
    assert 'vision_encoder' not in methods and not any(name.startswith('draft_') for name in methods)
    metadata = {}
    for name, plan in methods.items():
        if (name.startswith('get_') or name == 'has_vision_encoder') and len(plan.outputs) == 1:
            value = plan.values[plan.outputs[0]].val
            if isinstance(value, Int):
                metadata[name] = value.int_val
            elif isinstance(value, String):
                metadata[name] = value.string_val
            elif isinstance(value, Bool):
                metadata[name] = value.bool_val
    assert metadata['get_offgraph_kv_max_seqs'] == args.max_kv_seqs
    assert metadata['get_max_context_len'] == args.max_context and metadata['get_n_caches'] == 52
    assert metadata['get_activation_dtype'] in [int(ScalarType.BFLOAT16), 'bf16', 'bfloat16']
    assert metadata['get_max_prefill_chunk'] == args.max_step_tokens
    assert metadata['has_vision_encoder'] is False
    output = methods['forward_others'].values[methods['forward_others'].outputs[0]].val
    assert output.sizes == [64,202048]
    manifests = {}
    for name in forwards:
        for delegate in methods[name].delegates:
            for spec in delegate.compile_specs:
                if spec.key != 'offgraph_kv_manifest':
                    continue
                manifest = json.loads(bytes(spec.value))
                assert manifest['layout'] == 'batched-sequence' and manifest['maximum_capacity'] == args.max_context
                assert manifest['max_seqs'] == args.max_kv_seqs and manifest['max_segments'] == 2 * args.max_kv_seqs
                assert manifest['max_write'] == args.max_step_tokens and manifest['causal'] is True
                rings = [layer for layer in manifest['layers'] if layer['policy'] == 'ring']
                assert len(manifest['layers']) == 52 and len(rings) == 39
                assert all(layer['window'] == 2048 for layer in rings)
                manifests[name] = manifest
    assert set(manifests) == forwards
    builds = [json.loads(line) for line in (args.job_dir / 'actual-compiler-optimization.jsonl').read_text().splitlines()]
    optimized = [row for row in builds if row['optimization_flags'] and row['optimization_flags'][-1] == '-O2']
    assert optimized and not any(row['optimization_flags'] and row['optimization_flags'][-1] == '-O0' for row in builds)
    audit = dict(fresh_export=True, model='muse-glimmer', mode='solo, text only, CUDA batched-sequence offgraph KV',
                 selected_output_bound=output.sizes, constant_metadata=metadata, offgraph_manifests=manifests,
                 GGUF_sha256=digest(args.gguf), quant_source_sha256=sources,
                 attention_source_sha256=digest(kernels / 'sdpa_varlen.py'),
                 max_resident_sequences=args.max_kv_seqs, physical_columns=args.max_kv_seqs+1,
                 supported_prefix_entries_with_16_active=args.max_kv_seqs-16-1,
                 context=args.max_context, active_sessions=16, decode_ceiling=16, static_methods=[1,2,4,8,16,32,64],
                 AOTI_wrapper_optimization='O2', actual_optimized_builds=len(optimized),
                 artifacts={name:dict(sha256=digest(args.output / name),bytes=(args.output / name).stat().st_size)
                            for name in ['model.pte','aoti_cuda_blob.ptd']},
                 checkout_and_weight_source_unchanged=True, deployment_accepted=False,
                 performance_memory_and_GMRID_acceptance_pending=True)
    (args.output / 'artifact-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    state('artifact-exported-and-audited', supported_prefix_entries=args.max_kv_seqs-16-1)


if __name__ == '__main__':
    main()
