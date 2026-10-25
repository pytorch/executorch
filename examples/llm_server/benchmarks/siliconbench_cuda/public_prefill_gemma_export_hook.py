"""Opt-in export-process routing for two validated Gemma Q4 bulk GEMM shapes.

Never change an on-disk ET source file. Non-SM120 targets and a disabled
experiment retain the identical original dispatch object. Static forwards
remain outside this hook; only the existing dynamic bulk fallback is routed.
No new weight encoding, activation quantization, or static method is added.
"""
import hashlib
from pathlib import Path


DISPATCH_SHA256 = '1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'


def install_gemma_prefill_port(dispatch, *, capability, enabled=False):
    """Separate opt-in experiment for prepared metadata at Gate M128/M512 and Down M128.

    Keep the rejected MG experiment separate. All static methods and
    other dynamic M values retain their original operations.
    """
    original = dispatch._dequant_matmul
    receipt = dict(enabled=enabled, capability=list(capability), installed=False,
                   original_dispatch_identical=True, static_forwards_changed=False,
                   weight_encoding_changed=False, A16=True,
                   only_dynamic_bulk=True, candidate_sizes=[128,512],
                   temporary_metadata=True,persistent_extra_model_copy=False)
    if not enabled or tuple(capability) != (12,0):
        return receipt
    assert hashlib.sha256(Path(dispatch.__file__).read_bytes()).hexdigest() == DISPATCH_SHA256
    import torch
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import can_launch_triton
    from public_prefill_gemma_fp32_aoti import prepared_prefill_op

    def route(x, q, sc, ss, z, zs, group_size):
        shape = (q.shape[0], q.shape[1]*2)
        eager = not torch.compiler.is_compiling() and isinstance(x.shape[0],int)
        CUDA_trace = torch.compiler.is_compiling() or can_launch_triton(x)
        if (eager or shape not in {(30720,3840),(3840,15360)}
                or group_size != 32 or x.dtype != torch.bfloat16 or not CUDA_trace):
            return original(x,q,sc,ss,z,zs,group_size)
        return prepared_prefill_op(x,q,sc,ss,z,zs)

    dispatch._dequant_matmul = route
    receipt.update(installed=True, original_dispatch_identical=False,
                   original_source_SHA256=DISPATCH_SHA256,
                   validated_Q4_shapes=[[30720,3840],[3840,15360]],
                   other_shapes_and_static_M_use_original=True,
                   bulk_arithmetic='selected Gemma FP32 reconstruction, final BF16 operands without FMA',
                   optimized_sizes_by_shape={'GateUp':[128,512],'Down':[128]})
    return receipt
