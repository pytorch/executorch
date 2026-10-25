"""Check the public prefill hook through ET edge/CUDA delegate serialization.

This is a linear integration probe with real MG packed weights, not a model
or SiliconBench acceptance test. Preserve failed exports and binding errors.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--screen-results', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    report = dict(model_accepted=False, HTTP_accepted=False, KA_started=False,
                  cases=[], artifacts={}, controller_pid=os.getpid())

    def save(status, **fields):
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        (a.output / 'progress.json').write_text(json.dumps(dict(
            status=status, controller_pid=os.getpid(), **fields)) + '\n')

    save('waiting-GPU-lock')
    with a.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
        try:
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor-cache'),
                              TRITON_CACHE_DIR=str(a.output / 'triton-cache'),
                              TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', OMP_NUM_THREADS='2')
            sys.path.insert(0, str(a.et_source.parent))
            import numpy as np
            import torch
            from gguf import GGUFReader, GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default, fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch import int4_dispatch as dispatch
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
            from executorch.exir._serialize._program import deserialize_pte_binary
            from public_prefill_export_hook import install_bulk_prefill_port
            assert torch.cuda.get_device_capability() == (12, 0)
            torch.set_num_threads(2)
            original = dispatch._dequant_matmul
            report['SM80_fallback'] = install_bulk_prefill_port(dispatch, capability=(8, 0), enabled=True)
            assert dispatch._dequant_matmul is original
            assert 'public_grouped_w4a16_aoti' not in sys.modules
            report['disabled_fallback'] = install_bulk_prefill_port(dispatch, capability=(12, 0), enabled=False)
            assert dispatch._dequant_matmul is original
            report['enabled_hook'] = install_bulk_prefill_port(dispatch, capability=(12, 0), enabled=True)
            routed = dispatch._dequant_matmul
            options = CudaBackend.get_aoti_compile_options

            def tuned(cls, specs):
                opts = dict(options(specs))
                opts['aot_inductor.compile_wrapper_opt_level'] = 'O2'
                return opts

            CudaBackend.get_aoti_compile_options = classmethod(tuned)
            assert sha(a.gguf) == '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            screen = json.loads(a.screen_results.read_text())
            reader = GGUFReader(str(a.gguf))

            def portable(t):
                assert t.tensor_type == GGMLQuantizationType.Q4_K
                n, k = tuple(map(int, t.shape[::-1]))
                raw = torch.from_numpy(np.array(t.data, copy=True)).view(torch.uint8).reshape(n, -1)
                return to_default(t.name, ExportableGGUFTensor.from_raw(raw, 'q4_k', torch.bfloat16))

            class Linear(torch.nn.Module):
                def __init__(self, operation):
                    super().__init__()
                    self.operation = operation

                def forward(self, x, q, sc, ss, z, zs):
                    return self.operation(x, q, sc, ss, z, zs, 32)

            for role in ['ffn_gate_up', 'ffn_down']:
                if role == 'ffn_gate_up':
                    gate = next(t for t in reader.tensors if t.name == 'blk.0.ffn_gate.weight')
                    up = next(t for t in reader.tensors if t.name == 'blk.0.ffn_up.weight')
                    tensor = fuse_along_output([portable(gate), portable(up)])
                    n, k = 39936, 6656
                else:
                    t = next(t for t in reader.tensors if t.name.endswith('.ffn_down.weight')
                             and t.tensor_type == GGMLQuantizationType.Q4_K)
                    tensor = portable(t)
                    n, k = tuple(map(int, t.shape[::-1]))
                linear = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
                pack_linear_for_cuda(linear, {'weight': tensor})
                weights = tuple(getattr(linear.weight, key).cuda().contiguous()
                                for key in ['qdata', 'scale', 'scale_step', 'zero_point', 'zero_point_step'])
                packed = [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in weights]
                assert packed == screen['weight_sources'][role]['packed_SHA256']
                example = torch.randn((2048, k), device='cuda', dtype=torch.bfloat16)
                dynamic = ({0: torch.export.Dim('bulk_M', min=64, max=2048)}, {}, {}, {}, {}, {})
                for label, operation in [('original', original), ('public', routed)]:
                    save('exporting-ET-CUDA-delegate', role=role, arm=label)
                    ep = torch.export.export(Linear(operation), (example, *weights),
                                             dynamic_shapes=dynamic, strict=True)
                    assert len(ep.state_dict) == 0
                    graph = '\n'.join(str(m.graph) for m in ep.graph_module.modules()
                                      if isinstance(m, torch.fx.GraphModule))
                    (a.output / f'{role}-{label}-exported-graph.txt').write_text(graph)
                    if label == 'public':
                        assert 'cond' in graph and 'siliconbench.public_grouped_w4a16' in graph
                    specs = [CudaBackend.generate_method_name_compile_spec('forward')]
                    edge = to_edge_transform_and_lower(ep, partitioner=[CudaPartitioner(specs)],
                        compile_config=EdgeCompileConfig(_check_ir_validity=False))
                    executable = edge.to_executorch()
                    path = a.output / f'{role}-{label}.pte'
                    with path.open('wb') as f:
                        executable.write_to_file(f)
                    executable.write_tensor_data_to_file(str(a.output))
                    program = deserialize_pte_binary(path.read_bytes()).program
                    forward = next(plan for plan in program.execution_plan if plan.name == 'forward')
                    assert len(forward.delegates) == 1
                    assert forward.delegates[0].id == 'CudaBackend'
                    report['artifacts'][role+'-'+label] = dict(PTE_SHA256=sha(path), bytes=path.stat().st_size,
                        delegates=[d.id for d in forward.delegates], original_packed_SHA256=packed)
                    save('ET-CUDA-delegate-serialized', role=role, arm=label)
                del linear, weights, tensor, example, executable, edge, ep
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            report['ET_edge_and_CUDA_serialization_pass'] = True
            report['ET_runtime_and_model_numeric_validation_pending'] = True
            save('completed', artifacts=len(report['artifacts']))
        except Exception as e:
            (a.output / 'exception-traceback.txt').write_text(traceback.format_exc())
            save('failed', error_type=type(e).__name__, error=str(e)[-2000:])
            raise


if __name__ == '__main__':
    main()
