"""Export full Gemma48-layer independent KV only after mini qualification.

The receipt must qualify the mini storage mechanism and cold/steady performance;
this script refuses an unresolved regression or missing gate. Full-model export
is only preparation for full numerics, reclaim, same-GPU performance and F1.
Original GGUF, all48 layers, quantization, BF16 operands/KV, output64,
static1..64 and128K logical context remain unchanged. The prefill ceiling is
explicitly selected; 512 retains the existing export and 2048 restores upstream.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import urllib.request

GGUF_SHA256 = '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
GGUF_URL = ('https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/'
            'fc034cfff751157913579611efad8462ac1be606/gemma-4-12b-it-Q4_0.gguf')
PRECISION_SHA256 = '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-preparation-job', 'output', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True, help='Read-only identical main GGUF')
    parser.add_argument('--mini-qualification', type=Path, required=True)
    parser.add_argument('--runtime-native-job',type=Path,required=True)
    parser.add_argument('--layout',choices=['independent','global'],default='independent',
        help='Use global rows only for a separately retained same-GPU full baseline')
    parser.add_argument('--max-step-tokens',type=int,choices=[512,2048],default=512)
    parser.add_argument('--reuse-failed-job', type=Path,
        help='Copy compiler caches from a confirmed terminal attempt; preserve its records')
    parser.add_argument('--owned-GPU-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    from independent_session_full_native_binding import verify_qualification, alias_runtime
    qualification=verify_qualification(args.mini_qualification,'gemma4-first6',GGUF_SHA256,args.runtime_native_job)
    assert qualification['Gemma_source_preparation_SHA256']==digest(args.source_preparation_job/'source-preparation.json')

    if args.reuse_failed_job:
        from failed_export_cache_reuse import verify_failed_export
        verify_failed_export(args.reuse_failed_job, args.output,
            ('prepare_gemma_independent_session_full.py',))
        prior_gate = json.loads((args.reuse_failed_job/'native-gate.json').read_text())
        assert prior_gate['Gemma_source_preparation_SHA256'] == digest(args.source_preparation_job/'source-preparation.json')

    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_export_started=args.owned_GPU_child, candidate_accepted=False, **fields), indent=2) + '\n')
        pending.replace(args.output / 'progress.json')

    receipt_path = args.source_preparation_job / 'source-preparation.json'
    receipt = json.loads(receipt_path.read_text())
    original_gate = receipt['original_native_gate']
    source = Path(receipt['new_source'])
    assert receipt['original_source_and_binaries_unchanged']
    assert original_gate['native_GPU_tests_pass'] and original_gate['test_count'] == 74
    for name, expected in original_gate['binaries'].items():
        assert digest(name) == expected, name
    for name, expected in receipt['source_files'].items():
        assert digest(source / name) == expected, name
    shared = json.loads((package / 'independent_session_capacity_source.json').read_text())
    assert digest(package / 'independent_session_capacity_source.json') == original_gate['source_spec_SHA256']
    for name, record in shared['files'].items():
        if name != 'backends/cuda/triton/kernels/sdpa_varlen.py':
            assert digest(source / name) == record['after_SHA256'], name
    controls = {name: digest(source / name) for name in (
        'backends/cuda/triton/kernels/int4_quantized_gemm.py',
        'backends/cuda/triton/kernels/int5_quantized_gemm.py',
        'backends/cuda/triton/kernels/int6_quantized_gemm.py',
        'backends/cuda/autotune/launch_params.py',
        'extension/cuda/cuda_allocator.cpp')}
    if not args.owned_GPU_child:
        assert not args.output.exists(), 'Preserve previous attempts'
        args.output.mkdir(parents=True)
        state('verifying-qualified-Gemma-full-source-and-native-gates')
        (args.output/'mini-qualification-receipt.json').write_text(json.dumps(qualification,indent=2)+'\n')
        gguf = args.gguf or args.output / 'public-weights/gemma-4-12b-it-Q4_0.gguf'
        if args.gguf:
            assert digest(gguf) == GGUF_SHA256
        else:
            assert shutil.disk_usage(args.output).free > 20 * 2**30
            gguf.parent.mkdir()
            partial = gguf.with_suffix('.gguf.partial')
            state('downloading-pinned-public-Gemma-GGUF', public_download=GGUF_URL, bytes=0)
            total, previous = 0, 0
            with urllib.request.urlopen(GGUF_URL, timeout=120) as incoming, partial.open('xb') as outgoing:
                while chunk := incoming.read(8 * 1024 * 1024):
                    outgoing.write(chunk)
                    total += len(chunk)
                    if total - previous >= 256 * 2**20:
                        previous = total
                        state('downloading-pinned-public-Gemma-GGUF', public_download=GGUF_URL, bytes=total)
            assert digest(partial) == GGUF_SHA256, 'Public weights differ from main-comparison GGUF'
            partial.rename(gguf)
        (args.output / 'weight-source.json').write_text(json.dumps(dict(public_URL=GGUF_URL,
            GGUF=str(gguf), GGUF_SHA256=GGUF_SHA256, unchanged_quantization='Q4_0',
            matches_main_checkpoint=True), indent=2) + '\n')
        (args.output/'export-source-native-gate.json').write_text(json.dumps(original_gate,indent=2)+'\n')
        alias_runtime(args.runtime_native_job,package,args.output,source,
            {'Gemma_source_preparation_SHA256':digest(receipt_path)})
        state('waiting-GPU-lock', source=str(source))
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            state('exporting-Gemma48-independent-KV', source=str(source))
            with (args.output / 'export.log').open('x') as log:
                subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                    *sys.argv[1:], '--owned-GPU-child'], stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=21600)
        state('full-artifact-ready-pending-full-model-qualification', source=str(source), model='Gemma4-12B-48layers',
              runtime_native_gate_SHA256=digest(args.output / 'native-gate.json'),
              numerical_graph_memory_and_performance_validation_pending=True)
        return

    weights = json.loads((args.output / 'weight-source.json').read_text())
    gguf = Path(weights['GGUF'])
    assert digest(gguf) == GGUF_SHA256
    assert digest(package / 'gemma_runtime_precision.py') == PRECISION_SHA256
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
        TORCH_CUDA_ARCH_LIST='12.0', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
        OMP_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
    sys.path.insert(0, str(source.parent))
    import torch
    import triton
    import torch._inductor.cpp_builder as cpp
    from dataclasses import asdict, replace
    from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig, materialize_runtime_buffers
    from executorch.examples.models.gemma4_31b.batching_model import PackedGemma4
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import iter_gemma_gguf
    from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key
    from executorch.examples.models.gemma4_31b.cuda_packers import convert_quantized_tensors_for_cuda
    from executorch.examples.models.gemma4_31b.cuda_source_transformations import cuda_source_transformations
    from executorch.examples.models.gemma4_31b.export_batching import export_cuda_batching
    from executorch.extension.llm.export.load import load_checkpoint
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    import executorch.backends.cuda.quantize_op_dispatch.int4_dispatch as dispatch
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import chunked_dequant_linear
    from public_prefill_gemma_export_hook import install_gemma_prefill_port
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('qualified_Gemma_precision', package / 'gemma_runtime_precision.py')
    precision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precision)
    native.INT4_QUANTIZED_GEMM = precision.INT4_QUANTIZED_GEMM

    # Identical FP32 weight reconstruction and final BF16 operands to the
    # accepted Gemma runtime-column export. Both mini arms use the same hook.
    def dequant_linear(x, qdata, scale, scale_step, zero, zero_step, group_size):
        n, khalf = qdata.shape
        k = khalf * 2
        groups, gps = k // group_size, 256 // group_size
        def rows(i, j):
            packed = qdata[i:j].reshape(j-i, groups, group_size//2)
            q = torch.stack(((packed & 15).float(), (packed >> 4).float()), dim=-1).reshape(j-i, groups, group_size)
            s = scale[i:j].float() * scale_step[i:j].float().repeat_interleave(gps, dim=1)
            z = zero[i:j].float() * zero_step[i:j].float().repeat_interleave(gps, dim=1)
            dense = ((q-z[:, :, None])*s[:, :, None]).reshape(j-i, k).to(x.dtype)
            return torch.nn.functional.linear(x, dense)
        return chunked_dequant_linear(x, n, rows)
    dispatch._dequant_matmul = dequant_linear
    prefill = install_gemma_prefill_port(dispatch, capability=(12, 0), enabled=True, runtime_chunk_range=True)
    assert prefill['installed']
    full = Gemma4_31BConfig(hidden_size=3840, intermediate_size=15360,
        num_hidden_layers=48, num_attention_heads=16, num_key_value_heads=8,
        num_global_key_value_heads=1, max_seq_len=131072)
    config_receipt=dict(original_model_config=asdict(full),full_export_config=asdict(full),
        original_HF_config_SHA256='478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9',
        geometry_and_attention_rules_unchanged=True,retained_original_layer_indices=list(range(48)),
        mini_qualification_SHA256=digest(args.mini_qualification),full_model_qualified=False)
    assert len(full.layer_types)==48 and full.layer_types.count('sliding_attention')==40
    (args.output/'full-config-provenance.json').write_text(json.dumps(config_receipt,indent=2)+'\n')
    original_build = cpp.CppBuilder.build
    current_job = None
    def record_build(builder):
        flags = re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)', builder.get_command_line())
        with (current_job / 'actual-wrapper-optimization.jsonl').open('a') as log:
            log.write(json.dumps(dict(flags=flags, target=builder._name)) + '\n')
        return original_build(builder)
    cpp.CppBuilder.build = record_build
    snapshots = []
    for arm in (args.layout,):
        current_job = args.output / ('full-' + arm + '-export')
        current_job.mkdir()
        cache = current_job / 'cache'
        if args.reuse_failed_job:
            from failed_export_cache_reuse import copy_compiler_caches
            copy_compiler_caches(args.reuse_failed_job/('full-'+arm+'-export/cache'),
                cache,current_job/'reused-compiler-cache.json')
        (cache / 'host-zero-backed').mkdir(parents=True)
        os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
            TRITON_CACHE_DIR=str(cache / 'triton'), ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache / 'host-zero-backed'))
        artifact = args.output / ('full-' + arm + '-artifact')
        state('exporting-Gemma-full-' + arm, source=str(source))
        with torch.device('meta'):
            model = Gemma4_31B(full)
        load_checkpoint(str(gguf), model, key_map=gguf_to_model_key,
            tie_map={'embed_tokens.weight': ('lm_head.weight', True)},
            dtype=torch.bfloat16, raw_iter=iter_gemma_gguf)
        materialize_runtime_buffers(model, dtype=torch.bfloat16)
        convert_quantized_tensors_for_cuda(model)
        cuda_source_transformations(model, use_turboquant=False)
        model = PackedGemma4(model).eval()
        export_cuda_batching(model, str(artifact), max_step_tokens=args.max_step_tokens, max_kv_seqs=21,
            forward_sizes=(1, 2, 4, 8, 16, 32, 64), suppressed_token_ids=(258882, 258883),
            runtime_kv_columns=True, independent_capacity=arm == 'independent')
        del model
        from executorch.exir._serialize._program import deserialize_pte_binary
        from executorch.exir.schema import Int, String
        program = deserialize_pte_binary((artifact / 'model.pte').read_bytes()).program
        methods = {plan.name: plan for plan in program.execution_plan}
        metadata = {}
        for name, plan in methods.items():
            if name.startswith('get_') and len(plan.outputs) == 1:
                value = plan.values[plan.outputs[0]].val
                if isinstance(value, Int): metadata[name] = value.int_val
                elif isinstance(value, String): metadata[name] = value.string_val
        assert metadata['get_n_caches'] == 48 and metadata['get_vocab_size'] == 262144
        assert metadata['get_max_context_len'] == 131072 and metadata['get_max_prefill_chunk'] == args.max_step_tokens
        assert metadata['get_offgraph_kv_sequence_info_abi'] == (3 if arm == 'independent' else 2)
        forwards = {f'forward_{m}' for m in (1, 2, 4, 8, 16, 32, 64)} | {'forward_others'}
        assert forwards | {'sample', 'sample_argmax'} <= set(methods)
        assert not any(name.startswith(('vision_', 'draft_')) for name in methods)
        output = methods['forward_others'].values[methods['forward_others'].outputs[0]].val
        assert output.sizes == [64, 262144]
        manifests = {}
        for name in forwards:
            for delegate in methods[name].delegates:
                for record in delegate.compile_specs:
                    if record.key != 'offgraph_kv_manifest': continue
                    manifest = json.loads(bytes(record.value))
                    assert manifest.get('independent_capacity', False) == (arm == 'independent')
                    assert manifest['runtime_columns'] and manifest['max_seqs'] == 21
                    assert manifest['max_write'] == args.max_step_tokens and manifest['maximum_capacity'] == 131072
                    assert len(manifest['layers']) == 48
                    assert sum(layer['policy'] == 'ring' for layer in manifest['layers']) == 40
                    assert all(layer['window'] == 1024 for layer in manifest['layers'] if layer['policy'] == 'ring')
                    manifests[name] = manifest
        assert set(manifests) == forwards
        builds = [json.loads(line) for line in (current_job / 'actual-wrapper-optimization.jsonl').read_text().splitlines()]
        assert any(row['flags'] and row['flags'][-1] == '-O2' for row in builds)
        assert not any(row['flags'] and row['flags'][-1] == '-O0' for row in builds)
        audit = dict(model='Gemma4-12B-48layers', mini_is_not_full_model=False,
            retained_original_layer_indices=list(range(48)), independent_capacity=arm == 'independent',
            source_spec_sha256=original_gate['source_spec_SHA256'],
            native_gate_sha256=digest(args.output/'native-gate.json'),
            export_source_native_gate_sha256=digest(args.output/'export-source-native-gate.json'),
            runtime_native_source=str(args.runtime_native_job/'executorch'),
            Gemma_source_preparation_SHA256=digest(receipt_path),
            GGUF_sha256=GGUF_SHA256, precision_kernel_sha256=PRECISION_SHA256,
            logical_context=131072, initial_runtime_KV=512, safe_ring_rows=1024+args.max_step_tokens-1,
            max_step_tokens=args.max_step_tokens,
            declared_owner_limit=21, static_forwards=[1, 2, 4, 8, 16, 32, 64],
            metadata=metadata, offgraph_manifests=manifests, selected_logit_limit=64,
            prefill_fusion=prefill, bulk_prefill_A16_original_native_decode=True,
            AOTI_wrapper_optimization='O2', numerical_and_memory_acceptance_pending=True,
            candidate_accepted=False, artifacts={name:dict(sha256=digest(artifact/name),
                bytes=(artifact/name).stat().st_size) for name in ('model.pte', 'aoti_cuda_blob.ptd')})
        (artifact / 'independent-full-artifact-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        from executorch.backends.cuda.autotune import launch_params
        snapshots.append(dict(arm=arm, launch_picks={repr(k):v for k,v in launch_params._CACHE.items()}))
        (args.output / 'shared-autotune-cache-snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    assert snapshots[0]['launch_picks']
    assert len(snapshots)==1 and snapshots[0]['arm']==args.layout
    assert all(digest(source/name) == expected for name,expected in controls.items())
    assert all(digest(name) == expected for name,expected in original_gate['binaries'].items())
    (args.output / 'shared-autotune-cache-review.json').write_text(json.dumps(dict(
        existing_launch_picks_unchanged=True, unchanged_source_controls=controls,
        source_preparation_SHA256=digest(receipt_path), actual_compiled_math_review_pending=True,
        candidate_accepted=False), indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            output.mkdir(parents=True, exist_ok=True)
            failure = dict(status='failed', controller_pid=os.getpid(),
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                error=str(error), traceback=traceback.format_exc(), candidate_accepted=False,
                full_model_export_started='--owned-GPU-child' in sys.argv)
            (output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            (output/'progress.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
