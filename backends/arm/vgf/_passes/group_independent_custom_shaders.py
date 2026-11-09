# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.vgf.shaders.grid_sampler import CUSTOM_SHADER_DOMAIN_NAME
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.backend._ops import BackendOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class GroupIndependentCustomShadersPass(ArmPass):
    """Group independent Vulkan shaders before their TOSA postprocessing.

    Model Converter partitions operations in source order. An intervening TOSA
    operation can therefore split otherwise independent shaders' consumers
    across graph segments.

    Example:
        shader A
        TOSA graph work consuming A
        independent shader B

    can become:
        shader A
        independent shader B
        TOSA graph work consuming A

    The benefit is that when model-converter emits the VGF, the TOSA graph
    is not interleaved among shaders.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        """Schedule independent shaders together after TOSA lowering.

        Args:
            graph_module (GraphModule): Graph after TOSA operator lowering.

        Returns:
            PassResult: Graph with unchanged operators and updated node order.

        """
        graph = graph_module.graph
        modified = False
        for custom in list(graph.nodes):
            if (
                custom.target != exir_ops.backend.tosa.CUSTOM.default
                or custom.kwargs.get("domain_name") != CUSTOM_SHADER_DOMAIN_NAME
            ):
                continue
            inputs = set(custom.all_input_nodes)
            previous = custom.prev
            crossed_tosa = False
            while previous.op == "call_function":
                if previous in inputs:
                    break
                if previous.target == exir_ops.backend.tosa.CUSTOM.default:
                    if (
                        previous.kwargs.get("domain_name") == CUSTOM_SHADER_DOMAIN_NAME
                        and crossed_tosa
                    ):
                        previous.append(custom)
                        modified = True
                    break
                is_tosa = (
                    isinstance(previous.target, BackendOpOverload)
                    and previous.target.namespace == "tosa"
                )
                if (
                    not (is_tosa or previous.target == operator.getitem)
                    or previous.is_impure()
                ):
                    break
                crossed_tosa |= is_tosa
                previous = previous.prev

        if modified:
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
