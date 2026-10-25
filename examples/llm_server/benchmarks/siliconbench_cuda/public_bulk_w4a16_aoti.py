"""Keep the experimental prefill conditional inside one AOTI operator.

ET edge lowering must not require a portable slice_copy kernel for the
original branch's dequantization. Inductor lowers both branches at the
custom operator boundary; arithmetic and the original large branch remain
the same. This module is imported only by the enabled SM120 experiment.
"""
import torch
from torch.library import triton_op
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import (
    _dequant_matmul as original_dequant_matmul,
)
from public_grouped_w4a16_aoti import candidate_op


@triton_op('siliconbench::public_bulk_w4a16', mutates_args={})
def bulk_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
            ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor,
            config_index: int) -> torch.Tensor:
    return torch.cond(x.shape[0] <= 128,
        lambda a, b, c, d, e, f: candidate_op(a, b, c, d, e, f, config_index, False),
        lambda a, b, c, d, e, f: original_dequant_matmul(a, b, c, d, e, f, 32),
        (x, q, sc, ss, z, zs))


@bulk_op.register_fake
def fake(x, q, sc, ss, z, zs, config_index: int):
    return torch.empty((x.shape[0], q.shape[0]), dtype=x.dtype, device=x.device)
