# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import List, Optional, Tuple

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_dynamic_qdq,
    is_per_channel,
    is_qparam,
    is_quant,
)
from executorch.backends.xnnpack.utils.utils import (
    get_input_node,
    is_getitem,
    is_node,
    is_param_node,
)
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from torch.export import ExportedProgram


class GEMMConfig(XNNPartitionerConfig):
    """
    GEMM-like ops like Convolution, Addmm, Linear, mostly behave in the same way, in which we
    have some weight, bias, and activation node. The only difference between these types
    of ops are that the weight, bias, and activations are in different indicies of the
    nodes arguments, this class helps to generalize the logic needed to partition these
    different ops
    """

    def __init__(self, weight_idx, bias_idx, act_idx, fused_acts):
        super().__init__()
        self.weight_idx = weight_idx
        self.bias_idx = bias_idx
        self.act_idx = act_idx
        self.fused_acts = fused_acts

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            # short circuit if we don't pass common constraints
            return False

        precision = self._detect_precision(node)
        if precision not in self.enabled_precision_types:
            # detected precision but it is either disabled or not supported
            return False

        is_valid, _ = self.get_deps(node, ep, precision)
        return is_valid

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        partition = [node]
        precision = self._detect_precision(node)
        _, deps = self.get_deps(node, ep, precision)
        partition.extend(deps)

        return partition

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return None

    def _detect_precision(self, node: torch.fx.Node) -> ConfigPrecisionType:
        weight = get_input_node(node, self.weight_idx)

        if not is_dequant(weight):
            return ConfigPrecisionType.FP32

        activation = get_input_node(node, self.act_idx)
        if is_dynamic_qdq(activation):
            return ConfigPrecisionType.DYNAMIC_QUANT

        return ConfigPrecisionType.STATIC_QUANT

    def get_deps(
        self, node: torch.fx.Node, ep: ExportedProgram, precision: ConfigPrecisionType
    ) -> Tuple[bool, List[torch.fx.Node]]:
        """
        Gets all dependencies for this gemm partition. Returns a tuple of
        a bool indicating if the deps are valid and a list of all the
        dep nodes
        """
        valid_bias, bias_deps = self._get_bias_deps(node, ep, precision)
        valid_weight, weight_deps = self._get_weight_deps(node, ep, precision)
        valid_act, act_deps = self._get_act_deps(node, ep, precision)
        valid_output, output_deps = self._get_output_deps(node, ep, precision)

        valid_deps = valid_bias and valid_weight and valid_act and valid_output
        deps = list(chain(bias_deps, weight_deps, act_deps, output_deps))

        return valid_deps, deps

    def _get_weight_deps(
        self, node: torch.fx.Node, ep: ExportedProgram, precision: ConfigPrecisionType
    ) -> Tuple[bool, List[torch.fx.Node]]:
        gemm_deps = []
        if precision == ConfigPrecisionType.FP32:
            # First find the weight
            weight_node = get_input_node(node, self.weight_idx)
            if not is_param_node(ep, weight_node):
                return (False, [])  # weight must be a static param
            gemm_deps.append(weight_node)

            return (True, gemm_deps)
        else:
            # Quantized Weight deps
            dequant_node = get_input_node(node, self.weight_idx)
            if not is_dequant(dequant_node):
                return False, []
            gemm_deps.append(dequant_node)
            weight = get_input_node(dequant_node, 0)
            if not is_param_node(ep, weight):
                return False, []
            gemm_deps.append(weight)

            if is_per_channel(dequant_node):
                if len(dequant_node.all_input_nodes) < 2:
                    # Expected channel quantized to have scale/zp nodes
                    return False, []

                gemm_deps.extend(dequant_node.all_input_nodes[1:3])
            return (True, gemm_deps)

    def _get_output_deps(
        self, node: torch.fx.Node, ep: ExportedProgram, precision: ConfigPrecisionType
    ) -> Tuple[bool, List[torch.fx.Node]]:
        gemm_deps = []
        if precision == ConfigPrecisionType.STATIC_QUANT:
            # Look for fused activations and tail end quant node
            node_users = list(node.users.keys())
            if len(node_users) != 1:
                # Expect quantized node to have a single output (fused act or dequant)
                return False, []

            # Check if the quantized pattern has a fused activation
            n_output = node_users[0]
            if (
                n_output.op == "call_function"
                and format_target_name(n_output.target.__name__) in self.fused_acts
            ):
                gemm_deps.append(n_output)
                fused_out_users = list(n_output.users.keys())
                if len(fused_out_users) == 1:
                    n_output = fused_out_users[0]

            if not is_quant(n_output):
                # Expected gemm_node --> fused_act (optional) --> dequant
                return (False, [])
            gemm_deps.append(n_output)
        elif precision == ConfigPrecisionType.FP32:
            # Look for fused activations only, and partition with fp32 op
            node_users = list(node.users.keys())
            if len(node_users) == 1:
                n_output = node_users[0]
                if (
                    n_output.op == "call_function"
                    and format_target_name(n_output.target.__name__) in self.fused_acts
                ):
                    gemm_deps.append(n_output)

        # FP32 and Dynamic Quant have no output dependencies
        return (True, gemm_deps)

    def _get_bias_deps(
        self, node: torch.fx.Node, ep: ExportedProgram, precision: ConfigPrecisionType
    ) -> Tuple[bool, List[torch.fx.Node]]:
        gemm_deps = []
        if len(node.all_input_nodes) > 2:
            bias_node = get_input_node(node, self.bias_idx)
            if bias_node:
                if not is_param_node(ep, bias_node):
                    return (False, [])  # bias node must be a static param
                gemm_deps.append(bias_node)

        return (True, gemm_deps)

    def _get_act_deps(
        self, node: torch.fx.Node, ep: ExportedProgram, precision: ConfigPrecisionType
    ) -> Tuple[bool, List[torch.fx.Node]]:
        gemm_deps = []
        if precision == ConfigPrecisionType.FP32:
            return (True, [])
        else:
            dq_input = get_input_node(node, self.act_idx)
            if not is_dequant(dq_input):
                # Expected static quant input to be dequant node
                return False, []
            gemm_deps.append(dq_input)
            if precision == ConfigPrecisionType.STATIC_QUANT:
                # if static quant we are done after finding first dq_input
                return (True, gemm_deps)

            # q input node
            q_input = get_input_node(dq_input, 0)
            if not is_quant(q_input):
                return (False, [])

            gemm_deps.append(q_input)
            if not (is_node(q_input.args[1]) and is_node(q_input.args[2])):
                # expected to find getitem node from choose qparam
                return (False, [])

            getitem1 = get_input_node(q_input, 1)
            getitem2 = get_input_node(q_input, 2)

            if not (is_getitem(getitem1) and is_getitem(getitem2)):
                # expected getitem node from choose qparam
                return (False, [])

            gemm_deps.extend([getitem1, getitem2])
            choose_qparam = get_input_node(getitem1, 0)
            if not is_qparam(choose_qparam):
                # expected to find choose_qparam node
                return (False, [])
            gemm_deps.append(choose_qparam)
            return (True, gemm_deps)


class LinearConfig(GEMMConfig):
    target_name = "linear.default"

    def __init__(self):
        super().__init__(
            weight_idx=1,
            bias_idx=2,
            act_idx=0,
            fused_acts=["relu.default", "hardtanh.default"],
        )

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.linear.default

    def supported_precision_types(self):
        return [
            ConfigPrecisionType.DYNAMIC_QUANT,
            ConfigPrecisionType.FP32,
            ConfigPrecisionType.STATIC_QUANT,
        ]


class AddmmConfig(GEMMConfig):
    target_name = "addmm.default"

    def __init__(self):
        super().__init__(weight_idx=2, bias_idx=0, act_idx=1, fused_acts=[])

    def supported_precision_types(self):
        return [
            ConfigPrecisionType.FP32,
            ConfigPrecisionType.STATIC_QUANT,
        ]
