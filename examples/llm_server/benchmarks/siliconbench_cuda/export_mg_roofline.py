"""Export a separate MG artifact for the seven fixed-context roofline mixes.

Keep the source, weights, static methods and BF16 bulk activation policy.
Context before each measured step is1024. The native128K model context is retained; runtime limits each history
to2048 during this separate experiment. The512-token chunk appends after1K.
The artifact is a candidate; successful export is not server/F1/performance
acceptance. A CPU-only parent owns the GPU lock through child process teardown.
"""
import argparse
import fcntl
import hashlib
import importlib
import importlib.util
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
    parser.add_argument('--attention-source-sha256', required=True)
    parser.add_argument('--max-kv-seqs', type=int, choices=[32], default=32)
    parser.add_argument('--max-step-tokens', type=int, choices=[1024], default=1024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--accepted-ka-report',type=Path)
    parser.add_argument('--a100-control-sha256')
    parser.add_argument('--owned-gpu-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--max-context', type=int, choices=[131072], default=131072)
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
                      roofline_numerics_and_timing_acceptance_pending=True)
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
    assert digest(kernels / 'sdpa_varlen.py') == args.attention_source_sha256
    cache = args.job_dir / 'cache'
    (cache / 'host-zero-backed').mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
                      TORCH_CUDA_ARCH_LIST='12.0', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                      ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache / 'host-zero-backed'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                      OMP_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(args.et_source.parent))
    # Use the verified attention source from the selected MG main artifact.
    # The C checkout also serves Gemma and has different D128 routing. Load
    # this module before model imports without changing that checkout.
    attention_reference = Path(__file__).with_name('mg_main_sdpa_varlen_reference.py')
    assert digest(attention_reference) == '5ce65105639785e625739c90c0b208a77d12f8544208c85aa66e143d34a4e8ea'
    parent_name = 'executorch.backends.cuda.triton.kernels'
    module_name = parent_name + '.sdpa_varlen'
    parent = importlib.import_module(parent_name)
    assert module_name not in sys.modules, 'Load selected MG attention before any model imports'
    spec = importlib.util.spec_from_file_location(module_name, attention_reference)
    assert spec is not None and spec.loader is not None
    selected_attention = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = selected_attention
    spec.loader.exec_module(selected_attention)
    setattr(parent, 'sdpa_varlen', selected_attention)
    import torch
    import torch._inductor.cpp_builder as cpp
    from executorch.backends.cuda.cuda_backend import CudaBackend
    assert torch.cuda.get_device_capability() == (12,0)
    # Some preserved export checkouts set O0 inside their export entry point.
    # Apply the compiler option when each delegate is actually lowered, so the
    # AOTI cache key and actual wrapper compilation both use O2. This affects
    # only this isolated exporter process; framework sources remain unchanged.
    # Multimethod weight sharing bypasses preprocess(), so override the shared
    # compile-options method used by both lowering entry points.
    original_options = CudaBackend.get_aoti_compile_options

    def optimized_options(cls, compile_specs):
        options = dict(original_options(compile_specs))
        options["aot_inductor.compile_wrapper_opt_level"] = "O2"
        return options

    CudaBackend.get_aoti_compile_options = classmethod(optimized_options)
    ka_receipt=None
    if args.accepted_ka_report is not None:
        from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
        from ka_v8_export_hook import configure_mg_export
        native.INT4_QUANTIZED_GEMM,ka_receipt=configure_mg_export(
            native.INT4_QUANTIZED_GEMM,capability=torch.cuda.get_device_capability(),
            accepted_report=args.accepted_ka_report,candidate_source=Path(__file__).resolve().parent,
            a100_control_sha256=args.a100_control_sha256,job_dir=args.job_dir)
    original_build = cpp.CppBuilder.build

    def record_build(self):
        flags = re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)', self.get_command_line())
        with (args.job_dir / 'actual-compiler-optimization.jsonl').open('a') as log:
            log.write(json.dumps(dict(optimization_flags=flags, target_name=self._name,
                                      command=self.get_command_line())) + '\n')
        assert not flags or flags[-1] != '-O0', \
            'O0 compilation rejected before building; inspect the shared CUDA compile options'
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
                 attention_source_sha256=digest(attention_reference),
                 unchanged_checkout_attention_sha256=digest(kernels / 'sdpa_varlen.py'),
                 selected_MG_main_attention_loaded_before_model_imports=True,
                 max_resident_sequences=args.max_kv_seqs, physical_columns=args.max_kv_seqs+1,
                 prefix_caching=False, measured_context_before_step=1024,
                 context=args.max_context, active_sessions=32, decode_ceiling=32, static_methods=[1,2,4,8,16,32,64],
                 AOTI_wrapper_optimization='O2', actual_optimized_builds=len(optimized),
                 artifacts={name:dict(sha256=digest(args.output / name),bytes=(args.output / name).stat().st_size)
                            for name in ['model.pte','aoti_cuda_blob.ptd']},
                 checkout_and_weight_source_unchanged=True, deployment_accepted=False,
                 roofline_numerics_and_timing_acceptance_pending=True)
    if ka_receipt is not None:
        from ka_v8_export_hook import audit_route_log
        audit['KA_route_receipt']=audit_route_log(args.job_dir,ka_receipt)
    (args.output / 'artifact-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    state('artifact-exported-and-audited', runtime_sessions=32, prefix_caching=False, roofline_measurements_pending=True)


if __name__ == '__main__':
    main()
