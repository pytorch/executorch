"""Export an isolated original/public 4-layer MG prefill integration pair.

Use original GGUF layers 0..3 and complete embeddings/head. This is a
feasibility model; export success is never full-model quality/perf acceptance.
Both arms retain static methods, BF16, selected logits, and original packing.
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
import shutil
import subprocess
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--delegate-proof', type=Path, required=True)
    p.add_argument('--arm', choices=['original', 'public'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--seed-reference-export-cache', type=Path,
                   help='Copy a completed reference export cache into this new job; never modify the reference')
    p.add_argument('--owned-worker', action='store_true')
    a = p.parse_args()

    def state(status, **fields):
        (a.job / 'progress.json').write_text(json.dumps(dict(status=status,
            controller_pid=os.getpid(), arm=a.arm, model_accepted=False, **fields)) + '\n')

    if not a.owned_worker:
        assert not a.job.exists() and not a.output.exists()
        a.job.mkdir(parents=True)
        state('waiting-GPU-lock')
        with a.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader'], text=True).strip()
            (a.job / 'lock-owner.json').write_text(json.dumps({'pid': os.getpid()}))
            try:
                with (a.job / 'export.log').open('x') as f:
                    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                        *sys.argv[1:], '--owned-worker'], stdout=f, stderr=subprocess.STDOUT,
                        check=True, timeout=7200)
                assert json.loads((a.job / 'progress.json').read_text())['status'] == 'artifact-audited'
                state('completed', artifact=str(a.output), numerical_runtime_and_perf_pending=True)
            except Exception as e:
                state('failed', error=str(e)[-1500:])
                raise
        return
    assert json.loads((a.job / 'lock-owner.json').read_text())['pid'] == os.getppid()
    proof = json.loads(a.delegate_proof.read_text())
    assert proof['ET_edge_and_CUDA_serialization_pass'] and len(proof['artifacts']) == 4
    assert sha(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    cache = a.job / 'cache'
    (cache / 'host-zero-backed').mkdir(parents=True)
    if a.seed_reference_export_cache is not None:
        assert a.arm == 'public'
        reference = a.seed_reference_export_cache.resolve()
        assert reference != cache.resolve()
        assert json.loads((reference.parent / 'progress.json').read_text())['status'] == 'completed'
        for name in ['inductor', 'triton']:
            assert (reference / name).is_dir()
            shutil.copytree(reference / name, cache / name)
        (a.job / 'reference-cache-copy.json').write_text(json.dumps(dict(
            source=str(reference), source_modified=False, hardlinks=False,
            unchanged_kernel_cache_seed=True, numeric_gate_still_required=True))+'\n')
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
        TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_COMPILE_THREADS='1',
        ET_CUDA_HOST_ZERO_BACKING_DIR=str(cache / 'host-zero-backed'),
        TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='1', OMP_NUM_THREADS='2')
    sys.path.insert(0, str(a.et_source.parent))
    ref = Path(__file__).with_name('mg_main_sdpa_varlen_reference.py')
    assert sha(ref) == '5ce65105639785e625739c90c0b208a77d12f8544208c85aa66e143d34a4e8ea'
    parent_name = 'executorch.backends.cuda.triton.kernels'
    parent = importlib.import_module(parent_name)
    name = parent_name + '.sdpa_varlen'
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, ref)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(parent, 'sdpa_varlen', module)
    import torch
    import torch._inductor.cpp_builder as cpp
    from dataclasses import replace
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.quantize_op_dispatch import int4_dispatch as dispatch
    from executorch.examples.models.muse_glimmer.export import export_solo as exporter
    from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader as loader
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(2)
    source_sha = sha(dispatch.__file__)
    original_options = CudaBackend.get_aoti_compile_options

    def optimized(cls, specs):
        options = dict(original_options(specs))
        options['aot_inductor.compile_wrapper_opt_level'] = 'O2'
        return options

    CudaBackend.get_aoti_compile_options = classmethod(optimized)
    original_build = cpp.CppBuilder.build

    def build(builder):
        flags = re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)', builder.get_command_line())
        assert not flags or flags[-1] not in ['-O0', '-O1']
        with (a.job / 'actual-wrapper-optimization.jsonl').open('a') as f:
            f.write(json.dumps({'flags': flags, 'target': builder._name}) + '\n')
        return original_build(builder)

    cpp.CppBuilder.build = build
    if a.arm == 'public':
        from public_prefill_export_hook import install_bulk_prefill_port
        hook = install_bulk_prefill_port(dispatch, capability=(12, 0), enabled=True)
        assert hook['installed']
    else:
        hook = dict(installed=False)
        assert 'public_bulk_w4a16_aoti' not in sys.modules
    state('loading-original-GGUF')
    full = MuseGlimmerConfig(max_seq_len=131072)
    config = replace(full, n_layers=4)
    atomic = loader._atomic_sd_from_gguf(str(a.gguf), torch.bfloat16)
    retained = {}
    for name, value in atomic.items():
        match = re.match(r'^layers\.(\d+)\.', name)
        if match is None or int(match.group(1)) < 4:
            retained[name] = value
    del atomic
    model = loader._finalize(retained, 'cuda', config, torch.bfloat16)
    del retained
    # Real model export starts from CPU examples/FakeTensors. Installing a
    # hook is insufficient: require the candidate operators in the actual
    # strict exported graph before paying for AOTI compilation.
    # export_solo imports torch.export.export inside export_cuda_batching.
    # Patch the provider before that import, rather than a nonexistent module
    # attribute. The wrapper observes the real strict model export.
    original_export = torch.export.export
    graph_audits = []
    if a.arm == 'public':
        from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
        expected_routes = sum(isinstance(getattr(m, 'weight', None), CudaCoalescedInt4Tensor)
            and (m.weight.qdata.shape[0], m.weight.qdata.shape[1]*2) in {(39936,6656),(6656,19968)}
            for m in model.modules())
        assert expected_routes >= 4
    else:
        expected_routes = 0

    def traced_export(*args, **kwargs):
        program = original_export(*args, **kwargs)
        count = sum(node.op == 'call_function' and 'siliconbench.public_bulk_w4a16' in str(node.target)
            for module in program.graph_module.modules() if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes)
        item = dict(dynamic=kwargs.get('dynamic_shapes') is not None,
                    candidate_operator_calls=count,
                    example_devices=[str(t.device) for t in args[1] if isinstance(t, torch.Tensor)])
        graph_audits.append(item)
        (a.job / 'actual-strict-export-route-audit.json').write_text(json.dumps(graph_audits, indent=2)+'\n')
        if count:
            assert a.arm == 'public' and item['dynamic'] and count == expected_routes
        if a.arm == 'public' and args[0] is model and item['dynamic'] and args[1][0].ndim == 2 and args[1][0].shape[1] == 2048:
            assert count == expected_routes, 'Experimental prefill routing missing from actual CPU/FakeTensor model export'
        return program

    torch.export.export = traced_export
    state('exporting-mini')
    exporter.export_cuda_batching(model, config, str(a.output), max_step_tokens=2048,
        max_cells=65536, forward_sizes=[1, 2, 4, 8, 16, 32, 64],
        device_sampling=True, kv_layout='sequence', max_kv_seqs=32)
    from executorch.exir._serialize._program import deserialize_pte_binary
    program = deserialize_pte_binary((a.output / 'model.pte').read_bytes()).program
    methods = {plan.name: plan for plan in program.execution_plan}
    assert {f'forward_{m}' for m in [1, 2, 4, 8, 16, 32, 64]} | {'forward_others'} <= set(methods)
    out = methods['forward_others'].values[methods['forward_others'].outputs[0]].val
    assert out.sizes == [64, 202048]
    assert sha(dispatch.__file__) == source_sha
    assert sum(item['candidate_operator_calls'] for item in graph_audits) == expected_routes
    audit = dict(arm=a.arm, n_layers=4, miniature_not_full_MG=True,
        GGUF_SHA256=sha(a.gguf), original_dispatch_source_SHA256=source_sha,
        hook=hook, actual_export_graph_route_audit=graph_audits,
        actual_export_candidate_calls=expected_routes,
        reference_cache_copied=a.seed_reference_export_cache is not None,
        logical_context=131072, max_step_tokens=2048, max_kv_seqs=32,
        static_forwards=[1, 2, 4, 8, 16, 32, 64], selected_logits=out.sizes,
        runtime_numeric_and_perf_pending=True, full_model_or_HTTP_accepted=False,
        artifacts={n:dict(SHA256=sha(a.output / n), bytes=(a.output / n).stat().st_size)
                   for n in ['model.pte', 'aoti_cuda_blob.ptd']})
    (a.output / 'public-prefill-mini-artifact-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    state('artifact-audited')


if __name__ == '__main__':
    main()
