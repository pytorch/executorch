# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from typing import Optional

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack._passes.lift_constant_scalar_operands_pass import (
    LiftConstantScalarOperandsPass,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.partition.graphs import sdpa
from executorch.backends.xnnpack.utils.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConvertToSDPAPass(XNNPACKPass):
    def get_scale(self, match: InternalMatch) -> Optional[float]:
        """
        Return the SDPA scale recovered from the matched pre-QK^T multiplications.

        The multiplier may be a scalar literal or a constant tensor introduced by
        scalar lifting. The decomposition applies the square root of the attention
        scale before QK^T, so the extracted multiplier is squared to recover the
        original value.
        """
        for node in match.nodes_map.values():
            if node.op != "call_function" or node.target not in {
                exir_ops.edge.aten.mul.Scalar,
                exir_ops.edge.aten.mul.Tensor,
            }:
                continue

            scale = node.args[1]

            # Extract the scale from the constant tensor introduced by scalar
            # lifting.
            if node.target == exir_ops.edge.aten.mul.Tensor:
                if not isinstance(scale, torch.fx.Node):
                    continue
                scale_tensor = get_param_tensor(self.exported_program, scale)
                if scale_tensor is None or scale_tensor.numel() != 1:
                    continue
                scale = scale_tensor.item()

            dtype = torch.float
            mul_val = node.meta.get("val", None)
            if mul_val is not None:
                dtype = mul_val.dtype

            if isinstance(scale, float):
                # Convert scale value to fp16 (reducing precision)
                scale = torch.tensor(scale, dtype=dtype).item()

                # since scale we extracted this before the QK^T.
                return scale**2
        return None

    def assert_2d_mask(self, match: InternalMatch) -> None:
        """
        No better way to do this right now. Ideally we don't want to partition this.
        """
        mask = match.placeholder_nodes[-1]
        mask_shape = get_shape(mask)
        if len(mask_shape) != 2:
            raise Exception(f"Mask rank is not 2 got {mask_shape}")

    def create_sdpa(
        self,
        graph_module: torch.fx.GraphModule,
        match: InternalMatch,
    ):
        logger.debug(f"Matched Subgraph: {match}")

        scale = self.get_scale(match)
        assert scale is not None, "Could not find scale"
        logger.debug(f"scale: {scale}")

        self.assert_2d_mask(match)

        output = match.returning_nodes[0]

        with graph_module.graph.inserting_before(output):
            sdpa_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.scaled_dot_product_attention.default,  # HACK not edge_op/CATen
                tuple(match.placeholder_nodes),
                kwargs={"scale": scale},
            )

        sdpa_node.meta["val"] = sdpa_node.target(  # pyre-fixme[29]
            *[n.meta["val"] for n in match.placeholder_nodes],
            scale=scale,
        )

        logger.debug(
            f"Replacing {output}{get_shape(output)} node with {sdpa_node}{get_shape(sdpa_node)}"
        )
        output.replace_all_uses_with(sdpa_node)
        graph_module.graph.eliminate_dead_code()

    # override
    def call(self, graph_module: torch.fx.GraphModule):
        logger.debug("ConvertToSDPA Begin: ")
        logger.debug(graph_module.print_readable(print_output=False))

        for scalar_pattern in sdpa.get_graphs():
            # Deep-copy the cached scalar pattern so lifting it does not modify the
            # pattern used by non-lifted flows.
            tensor_pattern = deepcopy(scalar_pattern)
            tensor_pattern = LiftConstantScalarOperandsPass()(
                tensor_pattern
            ).graph_module

            for pattern in (scalar_pattern, tensor_pattern):
                sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
                matches = list(sm.match(graph_module.graph))
                for partition_to_replace in matches:
                    self.create_sdpa(graph_module, partition_to_replace)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        logger.debug("ConvertToSDPA End: ")
        logger.debug(graph_module.print_readable(print_output=False))

        return PassResult(graph_module, True)
