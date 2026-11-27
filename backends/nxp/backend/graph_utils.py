# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import _is_dequant_node
from executorch.backends.nxp.backend.ir.converter.quantization_utils import quantize
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

batch_norm_target_ops = [
    # Aten dialect variants
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    # Edge dialect variants
    exir_ops.edge.aten.batch_norm.default,
    exir_ops.edge.aten.native_batch_norm.default,
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
]


def is_op_node(node: Node, target_op) -> bool:
    if isinstance(target_op, list):
        target_ops = target_op
    else:
        target_ops = [target_op]

    return (
        node is not None
        and hasattr(node, "op")
        and node.op == "call_function"
        and hasattr(node, "target")
        and node.target in target_ops
    )


def is_batch_norm(node: Node) -> bool:
    return is_op_node(node, batch_norm_target_ops)


def get_output_shape(node: Node) -> tuple[torch.Size] | torch.Size | None:
    val = node.meta.get("val")

    if isinstance(val, torch.Tensor):
        return val.shape
    elif isinstance(val, tuple):
        return tuple([v.shape for v in val])

    return None


def is_clamp_preserved_under_quantization(
    node: Node, min_val: float = 0, max_val: float | None = None
) -> bool:
    """
    Checks if Clamp/ReLU/HardTanh is preserved under quantization and did
    not collapse into either identity or constant.

     Valid quant. bounds -                Quant. bounds -
    one hinge is preserved             Collapse to identity
            │   │                           │ │
            │   ▼/¯¯¯¯¯ ReLU6(x)            │ ▼/¯¯¯¯¯ ReLU6(x)
            │   /                           │ /
            │  /                            ▼/
            ▼ /                             /
        ¯¯¯¯¯ Hinge                   ¯¯¯¯¯ Hinge

        Args:
        node: Node to check whether is preserved
        min_val: Lower bound (hinge) of the operator (eg. 0 for ReLU)
        max_val: Upper bound of the operator (eg. 6 for ReLU6 or None for ReLU)
    """

    q_node = node.args[0]

    if not _is_dequant_node(q_node):
        return False

    if len(q_node.args) == 6:
        # per-tensor
        _, scale, zp, quant_min, quant_max, q_type = q_node.args
    else:
        # per-channel
        _, scale, zp, quant_min, quant_max, _, q_type = q_node.args

    quant_min = np.iinfo(q_type).min if quant_min is None else quant_min
    quant_max = np.iinfo(q_type).max if quant_max is None else quant_max

    q_type = torch_type_to_numpy_type(q_type).type
    quantized_min_val = quantize(
        value=min_val,
        zero_point=zp,
        scale=scale,
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=q_type,
    )

    if max_val is not None:
        quantized_max_val = quantize(
            value=max_val,
            zero_point=zp,
            scale=scale,
            quant_min=quant_min,
            quant_max=quant_max,
            dtype=q_type,
        )
        return (
            # If at least one bound is inside the quantization range
            # the hinge of the ReLU/HardTanh is preserved and therefore does not
            # collapse to identity or constant.
            (
                np.all(quant_min < quantized_min_val)
                or np.all(quantized_max_val < quant_max)
            )
            # When both operator bounds are outside the quantization range
            # the operator collapses into constant value (eg. 0 or 6 for ReLU6).
            and not np.all(quant_max < quantized_min_val)
            and not np.all(quant_min > quantized_max_val)
        )

    # Ensure ReLU/HardTanh hinge is preserved.
    return quant_min < quantized_min_val < quant_max
