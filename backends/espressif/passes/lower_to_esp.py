# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import torch
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.utils import get_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.fx import GraphModule

from .aten_to_esp_pass import _mul_kernel_args, _QParams


def _qparams(program, node, prefix):
    params = get_qparams_from_node(node, prefix)
    dtype = torch.int8 if prefix == "out" else torch.float32
    if (
        params is None
        or not params.is_per_tensor()
        or params.dtype != dtype
        or (params.quant_min, params.quant_max) != (-128, 127)
    ):
        return None
    scale = get_constant(program, params.scale)
    zero_point = get_constant(program, params.zero_point)
    if scale is None or zero_point is None:
        return None
    scale, zero_point = float(scale.item()), int(zero_point.item())
    if not math.isfinite(scale) or scale <= 0 or not -128 <= zero_point <= 127:
        return None
    return _QParams(scale, zero_point)


class LowerToEsp(ExportedProgramPassBase):
    """Lower supported fused-quant multiply operators to ESP-NN."""

    def call(self, exported_program):
        # The pass manager copies programs shallowly; decomposition also retires
        # qparam constants, so isolate both graph and constant dictionaries.
        graph = copy.deepcopy(exported_program.graph)
        for original, cloned in zip(exported_program.graph.nodes, graph.nodes):
            cloned.name = original.name
        module = GraphModule(exported_program.graph_module, graph)
        module.meta = exported_program.graph_module.meta.copy()
        exported_program._graph_module = module
        exported_program._graph_signature = copy.deepcopy(
            exported_program.graph_signature
        )
        exported_program._state_dict = exported_program.state_dict.copy()
        exported_program._constants = exported_program.constants.copy()

        modified = False
        for node in list(graph.nodes):
            if node.target != exir_ops.edge.fused_quant.mul.default:
                continue
            params = [
                _qparams(exported_program, node, prefix)
                for prefix in ("inp", "other", "out")
            ]
            if any(param is None for param in params):
                continue
            args = _mul_kernel_args(
                node.args[:2],
                params,
                node.meta.get("val"),
            )
            if args is None:
                continue
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    exir_ops.edge.esp_nn.quantized_mul_int8.default,
                    args,
                )
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
