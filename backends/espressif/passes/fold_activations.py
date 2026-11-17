# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import replace

import torch
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.backends.transforms.utils import get_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult


class FoldActivations(ExportedProgramPassBase):
    """Fold grouped clamps into the producer's output quantization."""

    def call(self, exported_program):
        graph = exported_program.graph
        modified = False
        for activation in list(graph.nodes):
            if activation.target not in (
                exir_ops.edge.fused_quant.relu.default,
                exir_ops.edge.fused_quant.hard_tanh.default,
            ):
                continue
            bounds = (
                (get_arg(activation, "min_val"), get_arg(activation, "max_val"))
                if activation.target == exir_ops.edge.fused_quant.hard_tanh.default
                else (0.0, float("inf"))
            )
            if not all(isinstance(bound, (int, float)) for bound in bounds) or not (
                bounds[0] <= bounds[1]
            ):
                continue
            producer = get_arg(activation, "inp")
            if (
                producer.target
                not in (
                    exir_ops.edge.fused_quant.add.default,
                    exir_ops.edge.fused_quant.sub.default,
                    exir_ops.edge.fused_quant.mul.default,
                    exir_ops.edge.fused_quant.linear.default,
                    exir_ops.edge.fused_quant.convolution_channels_last.default,
                )
                or len(producer.users) != 1
                or get_qparams_from_node(producer, "out") is not None
                or get_qparams_from_node(activation, "inp") is not None
            ):
                continue
            params = get_qparams_from_node(activation, "out")
            if (
                params is None
                or not params.is_per_tensor()
                or params.dtype != torch.int8
                or (params.quant_min, params.quant_max) != (-128, 127)
            ):
                continue
            scale = get_constant(exported_program, params.scale)
            zero_point = get_constant(exported_program, params.zero_point)
            if (
                scale is None
                or zero_point is None
                or not math.isfinite(scale.item())
                or scale.item() <= 0
                or not -128 <= zero_point.item() <= 127
            ):
                continue
            concrete = replace(params, scale=scale, zero_point=zero_point)
            lower, upper = concrete.quantize(
                torch.tensor(bounds, dtype=producer.meta["val"].dtype)
            ).tolist()
            for field in ("scale", "zero_point", "dtype"):
                set_arg(producer, f"out_{field}", get_arg(activation, f"out_{field}"))
            set_arg(producer, "out_quant_min", lower)
            set_arg(producer, "out_quant_max", upper)
            producer.meta["val"] = activation.meta["val"]
            activation.replace_all_uses_with(producer)
            graph.erase_node(activation)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
