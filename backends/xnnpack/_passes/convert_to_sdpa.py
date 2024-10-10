# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
from executorch.backends.transforms import get_shape

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.partition.graphs import sdpa
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConvertToSDPAPass(XNNPACKPass):
    def get_scale(self, match: InternalMatch) -> Optional[float]:
        """
        Returns the scale of the SDPA op.

        Scale: Optional[float] doesn't change the graph pattern.
        The default value can be calulated however we need to extract
        it for lowering when it is the user supplied value anyway.
        """
        for node in match.nodes_map.values():
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.mul.Scalar
            ):
                scale = node.args[1]

                dtype = torch.float
                mul_val = node.meta.get("val", None)
                if mul_val is not None:
                    dtype = mul_val.dtype

                if isinstance(scale, float):
                    # Convert scale value to fp16 (reducing precision)
                    scale = torch.tensor(scale, dtype=dtype).item()

                    # since scale we extracted this before the QK^T.
                    return scale**2
                break
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

        for pattern in sdpa.get_graphs():
            sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
            matches = list(sm.match(graph_module.graph))
            for partition_to_replace in matches:
                self.create_sdpa(graph_module, partition_to_replace)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        logger.debug("ConvertToSDPA End: ")
        logger.debug(graph_module.print_readable(print_output=False))

        return PassResult(graph_module, True)
