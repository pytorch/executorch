"""Opt-in export-process routing for two validated MG Q4 bulk GEMM shapes.

Never change an on-disk ET source file. Non-SM120 targets and a disabled
experiment retain the identical original dispatch object. Static forwards
remain outside this hook; only the existing dynamic bulk fallback is routed.
No new weight encoding, activation quantization, or static method is added.
"""
import hashlib
from pathlib import Path


DISPATCH_SHA256 = '1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'


def install_bulk_prefill_port(dispatch, *, capability, enabled=False):
    original = dispatch._dequant_matmul
    receipt = dict(enabled=enabled, capability=list(capability), installed=False,
                   original_dispatch_identical=True, static_forwards_changed=False,
                   weight_encoding_changed=False, A16=True,
                   only_dynamic_bulk=True, small_bulk_limit=128)
    if not enabled or tuple(capability) != (12, 0):
        return receipt
    assert hashlib.sha256(Path(dispatch.__file__).read_bytes()).hexdigest() == DISPATCH_SHA256
    import torch
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import can_launch_triton
    from public_grouped_w4a16 import Config
    from public_grouped_w4a16_aoti import candidate_op, config_index
    from public_bulk_w4a16_aoti import bulk_op
    indices = {
        (39936, 6656): config_index(Config(bm=32, bn=64, bk=128, split=1,
                                         warps=4, stages=3, group_m=8)),
        (6656, 19968): config_index(Config(bm=32, bn=64, bk=128, split=8,
                                         warps=4, stages=2, group_m=8)),
    }

    def route(x, q, sc, ss, z, zs, group_size):
        shape = (q.shape[0], q.shape[1] * 2)
        # Dynamo models symbolic sizes as Python ints. An isinstance(int)
        # guard alone silently excludes strict dynamic export. Ordinary eager
        # concrete calls remain original; quantized_linear already selects
        # native family ops before reaching this bulk fallback in static
        # methods. torch.cond preserves the symbolic runtime-M predicate.
        concrete_eager = not torch.compiler.is_compiling() and isinstance(x.shape[0], int)
        if (concrete_eager or shape not in indices or group_size != 32
                or x.dtype != torch.bfloat16 or not can_launch_triton(x)):
            return original(x, q, sc, ss, z, zs, group_size)
        index = indices[shape]
        return bulk_op(x, q, sc, ss, z, zs, index)

    dispatch._dequant_matmul = route
    receipt.update(installed=True, original_dispatch_identical=False,
                   original_source_SHA256=DISPATCH_SHA256,
                   validated_Q4_shapes=[list(s) for s in indices],
                   other_shapes_and_static_M_use_original=True,
                   bulk_arithmetic='staged BF16 reconstruction without FMA')
    return receipt
