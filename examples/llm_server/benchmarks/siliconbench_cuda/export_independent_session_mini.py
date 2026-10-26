"""Export only matched four-layer GGUF mini models after native KV tests pass."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--native-gate', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--role', choices=['mini-global', 'mini-independent'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--owned-GPU-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec_path = package / 'independent_session_capacity_source.json'
    spec = json.loads(spec_path.read_text())
    build = json.loads(args.native_gate.read_text())
    assert build['native_GPU_tests_pass'] and build['test_count'] >= 29
    assert len(build['independent_session_tests']) == 7
    assert build['source_spec_SHA256'] == digest(spec_path)
    assert Path(build['source']).resolve() == args.source.resolve()
    for name, record in spec['files'].items():
        assert digest(args.source / name) == record['after_SHA256'], name
    for name, expected in build['unchanged_controls'].items():
        assert digest(args.source / name) == expected, name

    def state(status, **fields):
        temp = args.job_dir / 'progress.pending.json'
        temp.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                       role=args.role, candidate_accepted=False,
                                       **fields), indent=2) + '\n')
        temp.replace(args.job_dir / 'progress.json')

    if not args.owned_GPU_child:
        assert not args.job_dir.exists() and not args.output.exists(), 'Preserve prior attempts'
        args.job_dir.mkdir(parents=True)
        state('waiting-GPU-lock')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            (args.job_dir / 'GPU-lock-owner.json').write_text(json.dumps({'pid': os.getpid()}) + '\n')
            try:
                with (args.job_dir / 'export.log').open('x') as log:
                    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                    *sys.argv[1:], '--owned-GPU-child'],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=21600)
                child = json.loads((args.job_dir / 'progress.json').read_text())
                assert child['status'] == 'artifact-exported-and-audited'
                deadline = time.monotonic() + 120
                while subprocess.check_output(
                        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    assert time.monotonic() < deadline, 'CUDA context teardown remains incomplete'
                    time.sleep(1)
                state('completed', artifact=str(args.output),
                      source_spec_sha256=digest(spec_path),
                      numerical_and_memory_acceptance_pending=True)
            except Exception as error:
                state('failed', error_type=type(error).__name__, detail=str(error)[-1600:])
                raise
        return

    assert json.loads((args.job_dir / 'GPU-lock-owner.json').read_text())['pid'] == os.getppid()
    with args.lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Export requires its parent to own the benchmark lock')
    assert not args.output.exists()
    assert digest(args.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    cache = args.job_dir / 'cache'
    (cache / 'host-zero-backed').mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
                      TORCH_CUDA_ARCH_LIST='12.0', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                      TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                      TRITON_CACHE_DIR=str(cache / 'triton'),
                      ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache / 'host-zero-backed'),
                      OMP_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
    sys.path.insert(0, str(args.source.parent))
    import torch
    import torch._inductor.cpp_builder as cpp
    from executorch.backends.cuda.cuda_backend import CudaBackend

    # Match the accepted main artifact's actual wrapper optimization. This
    # shared entry covers both CUDA lowering paths and updates the cache key;
    # editing a generated CppBuilder command would miss later compilation.
    original_options = CudaBackend.get_aoti_compile_options

    def optimized_options(cls, compile_specs):
        options = dict(original_options(compile_specs))
        options['aot_inductor.compile_wrapper_opt_level'] = 'O2'
        return options

    CudaBackend.get_aoti_compile_options = classmethod(optimized_options)
    from executorch.examples.models.muse_glimmer.export import export_solo as exporter
    from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader as loader
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig

    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(2)
    baseline_build = cpp.CppBuilder.build

    def record_build(builder):
        flags = re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)', builder.get_command_line())
        with (args.job_dir / 'actual-wrapper-optimization.jsonl').open('a') as log:
            log.write(json.dumps({'flags': flags, 'target': builder._name}) + '\n')
        return baseline_build(builder)

    cpp.CppBuilder.build = record_build
    state('exporting')
    from dataclasses import replace
    full_config = MuseGlimmerConfig(max_seq_len=131072)
    config = replace(full_config, n_layers=4)
    assert all(config.layer_use_rope(i) == full_config.layer_use_rope(i) and
               config.layer_window_size(i) == full_config.layer_window_size(i)
               for i in range(4))
    atomic = loader._atomic_sd_from_gguf(str(args.gguf), torch.bfloat16)
    retained = {}
    for name, value in atomic.items():
        match = re.match(r'^layers\.(\d+)\.', name)
        if match is None or int(match.group(1)) < 4:
            retained[name] = value
    del atomic
    model = loader._finalize(retained, 'cuda', config, torch.bfloat16)
    del retained
    independent = args.role == 'mini-independent'
    runtime_columns = True
    exporter.export_cuda_batching(
        model, config, str(args.output), max_step_tokens=2048, max_cells=65536,
        forward_sizes=[1, 2, 4, 8, 16, 32, 64], device_sampling=True,
        kv_layout='sequence', max_kv_seqs=21, runtime_kv_columns=True,
        independent_capacity=independent)
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import Int, String
    program = deserialize_pte_binary((args.output / 'model.pte').read_bytes()).program
    methods = {plan.name: plan for plan in program.execution_plan}
    assert {f'forward_{n}' for n in [1, 2, 4, 8, 16, 32, 64]} | {'forward_others', 'sample', 'sample_argmax'} <= set(methods)
    assert not any(n.startswith(('draft_', 'vision_')) for n in methods)
    metadata = {}
    for name, plan in methods.items():
        if name.startswith('get_') and len(plan.outputs) == 1:
            value = plan.values[plan.outputs[0]].val
            if isinstance(value, Int):
                metadata[name] = value.int_val
            elif isinstance(value, String):
                metadata[name] = value.string_val
    assert metadata['get_offgraph_kv_max_seqs'] == 21
    assert metadata['get_max_context_len'] == 131072
    assert metadata['get_n_caches'] == config.n_layers
    assert metadata['get_max_prefill_chunk'] == 2048
    assert metadata.get('get_offgraph_kv_sequence_info_abi', 1) == (3 if independent else 2)
    selected_output = methods['forward_others'].values[
        methods['forward_others'].outputs[0]].val
    assert selected_output.sizes == [64, 202048]
    manifests = {}
    for name in [f'forward_{n}' for n in [1, 2, 4, 8, 16, 32, 64]] + ['forward_others']:
        for delegate in methods[name].delegates:
            for compile_spec in delegate.compile_specs:
                if compile_spec.key != 'offgraph_kv_manifest':
                    continue
                manifest = json.loads(bytes(compile_spec.value))
                assert manifest['layout'] == 'batched-sequence'
                assert manifest['maximum_capacity'] == 131072
                assert manifest['max_seqs'] == 21 and manifest['max_segments'] == 42
                assert manifest['max_write'] == 2048 and len(manifest['layers']) == config.n_layers
                assert manifest['version'] == (3 if independent else 2)
                assert manifest.get('runtime_columns', False) == runtime_columns
                assert manifest.get('independent_capacity', False) == independent
                manifests[name] = manifest
    assert len(manifests) == 8
    builds = [json.loads(line) for line in (args.job_dir / 'actual-wrapper-optimization.jsonl').read_text().splitlines()]
    optimized = [row for row in builds if row['flags'] and row['flags'][-1] == '-O2']
    # Torch/Triton CPU helpers keep their ordinary O3. Every actual AOTI
    # wrapper is O2; no O0 wrapper may enter a numerical/performance pair.
    assert len(optimized) >= 10
    assert not any(row['flags'] and row['flags'][-1] == '-O0' for row in builds)
    audit = dict(role=args.role, source_spec_sha256=digest(spec_path),
                 GGUF_sha256=digest(args.gguf), retained_original_layer_indices=list(range(config.n_layers)),
                 mini_is_not_full_model=args.role.startswith('mini-'),
                 unchanged_quant_source=build['unchanged_controls'],
                 logical_context=131072, initial_runtime_KV=512, declared_owner_limit=21,
                 static_forwards=[1, 2, 4, 8, 16, 32, 64], runtime_columns=runtime_columns,
                 metadata=metadata, methods=sorted(methods),
                 selected_logit_limit=selected_output.sizes[0], offgraph_manifests=manifests,
                 CUDA_graphs_for_static_runtime=True,
                 bulk_prefill_A16_original_native_decode=True, AOTI_wrapper_optimization='O2',
                 actual_optimized_builds=len(optimized), O3_helpers_allowed=True,
                 artifacts={name:dict(sha256=digest(args.output / name), bytes=(args.output / name).stat().st_size)
                            for name in ['model.pte', 'aoti_cuda_blob.ptd']},
                 native_gate_sha256=digest(args.native_gate),
                 independent_capacity=independent,
                 numerical_and_memory_acceptance_pending=True, candidate_accepted=False)
    (args.output / 'independent-mini-artifact-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    state('artifact-exported-and-audited', artifact=str(args.output))


if __name__ == '__main__':
    main()
