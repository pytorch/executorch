# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, List, Set

import torch
from executorch.backends.qualcomm._passes.utils import get_runtime_inputs
from executorch.backends.qualcomm.partition.utils import (
    generate_qnn_executorch_option,
    is_htp_fp16,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_FALLBACK_NODE,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.experimental.symbolic_shapes import statically_known_true

logger = logging.getLogger(__name__)

_SIN_COS_OPS = {exir_ops.edge.aten.sin.default, exir_ops.edge.aten.cos.default}

# The largest rotary tables of the models this targets, at the 1024 token
# context of the Qualcomm LLM examples, have 1024 positions times a head size
# of up to 256. A sin or cos with a larger angle op, like Fourier features of
# a large input, stays delegated, so that no large computation moves to CPU.
_MAX_NUMEL = 1024 * 256

# Ops that only move or cast values. An angle that is a graph input moved this
# way alone is not computed by the graph, so the caller can keep it small.
_VIEW_OPS = {
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.expand_copy.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.select_copy.int,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
}

# The cheap elementwise and shape ops a positional angle, like t * freqs,
# positions / 10000 ** (2 * arange(n) / d) or t * exp(-log(10000) * arange(n)
# / n), is built from. where comes from a decomposed linspace, and full and
# cumsum from float positions like ones(n).cumsum(0).
_ANGLE_OPS = _VIEW_OPS | {
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.cumsum.default,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.log.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.pow.Tensor_Tensor,
    exir_ops.edge.aten.reciprocal.default,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.where.self,
}

# Positions are also built from masks, like attention_mask.cumsum(-1),
# mask.sum(-1) + ones(n).cumsum(0) or arange(n) < k.
_POSITION_OPS = _ANGLE_OPS | {
    exir_ops.edge.aten.sum.dim_IntList,
    exir_ops.edge.aten.eq.Scalar,
    exir_ops.edge.aten.eq.Tensor,
    exir_ops.edge.aten.ge.Scalar,
    exir_ops.edge.aten.ge.Tensor,
    exir_ops.edge.aten.gt.Scalar,
    exir_ops.edge.aten.gt.Tensor,
    exir_ops.edge.aten.le.Scalar,
    exir_ops.edge.aten.le.Tensor,
    exir_ops.edge.aten.lt.Scalar,
    exir_ops.edge.aten.lt.Tensor,
    exir_ops.edge.aten.ne.Scalar,
    exir_ops.edge.aten.ne.Tensor,
}


def _input_reads(node: torch.fx.Node) -> List[torch.fx.Node]:
    # Unlike all_input_nodes, keeps repeats, as in cat([t, t]).
    reads = []
    torch.fx.node.map_arg((node.args, node.kwargs), reads.append)
    return reads


def _is_exact(node: torch.fx.Node) -> bool:
    val = node.meta.get("val")
    return isinstance(val, torch.SymInt) or (
        isinstance(val, torch.Tensor) and not val.is_floating_point()
    )


class TagPositionalSinCos(ExportPass):
    """
    On HTP in float16 precision, tag the unquantized float32 sin and cos nodes
    whose angle the graph computes from its inputs, constants and integer
    positions with only the ops in _ANGLE_OPS, as a timestep or rotary
    embedding does, and that are larger than the graph inputs, and the
    integers computed from one, that their float ops read, taken together.
    Their angle ops are tagged too, so that the partitioner leaves them all
    on CPU. Fourier features of a graph input, like sin(x[..., None] *
    freqs), qualify too. A sin or cos is left alone when it, or an op of its
    angle, has more than _MAX_NUMEL elements. The limit holds for each sin or
    cos on its own, not for all the tagged ops together. A dynamic size
    counts with its largest value, so one without a max is left alone too.

    float16 cannot hold such an angle precisely: its values are 1.0 apart from
    1024 to 2048, where the angles of a timestep embedding can reach, and HTP's
    float16 Sin and Cos lose more on top of that for large angles. On CPU
    these nodes compute in float32, as the model does. A sin or cos whose angle
    reads an activation, even through integers, is left alone, so no conv or
    linear moves to CPU and the delegate is not split.

    The tagged ops then need CPU kernels at runtime. To delegate them as
    before, take passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
    and set passes_job[TagPositionalSinCos][QCOM_PASS_ACTIVATE_KEY] = False.
    """

    def __init__(self, edge_program, compiler_specs=None):
        super(TagPositionalSinCos, self).__init__()
        self.edge_program = edge_program
        self.htp_fp16 = False
        if compiler_specs is None:
            logger.debug(
                "TagPositionalSinCos does nothing, since it got no compiler_specs."
            )
        else:
            self.htp_fp16 = is_htp_fp16(
                flatbuffer_to_option(
                    generate_qnn_executorch_option(compiler_specs)
                ).backend_options
            )

    def call(self, graph_module: torch.fx.GraphModule):
        if not self.htp_fp16:
            return PassResult(graph_module, False)
        self.runtime_inputs = get_runtime_inputs(self.edge_program)
        nodes, positions = set(), {}
        for node in graph_module.graph.nodes:
            if (
                node.target in _SIN_COS_OPS
                and QCOM_QUANT_ATTRS not in node.meta
                and node.meta["val"].dtype == torch.float32
            ):
                angle_nodes = self._get_angle_nodes(node.args[0], positions)
                if not any(
                    n.target not in _VIEW_OPS for n in angle_nodes
                ) or not self._broadcasts_inputs(node, angle_nodes, positions):
                    continue
                if not all(
                    statically_known_true(n.meta["val"].numel() <= _MAX_NUMEL)
                    for n in angle_nodes | {node}
                    if isinstance(n.meta["val"], torch.Tensor)
                ):
                    logger.debug(
                        f"TagPositionalSinCos leaves {node.name} delegated, since "
                        f"it or its angle can have more than {_MAX_NUMEL} "
                        "elements. A dynamic size needs a max to be tagged."
                    )
                    continue
                nodes |= angle_nodes | {node}
        for node in nodes:
            node.meta[QCOM_FALLBACK_NODE] = True
        if nodes:
            logger.warning(
                f"TagPositionalSinCos keeps {len(nodes)} ops of positional sin "
                "and cos on CPU in float32, which needs their CPU kernels. "
                "To delegate them, set passes_job[TagPositionalSinCos]"
                "[QCOM_PASS_ACTIVATE_KEY] = False in the passes_job from "
                "get_qnn_pass_manager_cls().get_capture_program_passes()."
            )
        return PassResult(graph_module, bool(nodes))

    def _broadcasts_inputs(
        self,
        node: torch.fx.Node,
        angle_nodes: Set[torch.fx.Node],
        positions: Dict[torch.fx.Node, Set[torch.fx.Node]],
    ) -> bool:
        # A positional angle spreads a few positions or timesteps over a
        # frequency axis, so it is larger than the user inputs, and the
        # integers computed from one, that its float ops read, all together.
        # An angle no larger, like an image times a scale or a cat of
        # inputs, stays delegated. Integers built only from constants and
        # sizes, like the arange of a linspace, and the masks that integer
        # positions read do not count.
        numels = [
            arg.meta["val"].numel()
            for angle_node in angle_nodes | {node}
            if not _is_exact(angle_node)
            for arg in _input_reads(angle_node)
            if isinstance(arg.meta.get("val"), torch.Tensor)
            and (
                arg.name in self.runtime_inputs
                or (
                    _is_exact(arg)
                    and self._reads_runtime_input(positions.get(arg, set()))
                )
            )
        ]
        return not numels or statically_known_true(
            sum(numels) < node.meta["val"].numel()
        )

    def _reads_runtime_input(self, nodes: Set[torch.fx.Node]) -> bool:
        # An int input, like a size, only sets how many positions there are.
        return any(
            arg.name in self.runtime_inputs
            and isinstance(arg.meta.get("val"), torch.Tensor)
            for node in nodes
            for arg in node.all_input_nodes
        )

    def _get_angle_nodes(
        self, angle: torch.fx.Node, positions: Dict[torch.fx.Node, Set[torch.fx.Node]]
    ) -> Set[torch.fx.Node]:
        """
        The nodes computing `angle` from graph inputs, constants and integer
        positions, or an empty set if another op takes part.
        """
        nodes, stack = set(), [angle]
        while stack:
            node = stack.pop()
            if node in nodes or node.op != "call_function":
                continue
            if _is_exact(node):
                position_nodes = self._get_position_nodes(node, positions)
                if not position_nodes and not isinstance(
                    node.meta["val"], torch.SymInt
                ):
                    return set()
                nodes |= position_nodes
                continue
            if node.target not in _ANGLE_OPS or QCOM_QUANT_ATTRS in node.meta:
                return set()
            nodes.add(node)
            stack.extend(node.all_input_nodes)
        return nodes

    def _get_position_nodes(
        self, node: torch.fx.Node, positions: Dict[torch.fx.Node, Set[torch.fx.Node]]
    ) -> Set[torch.fx.Node]:
        """
        `node` and the integer nodes it is computed from, if they are built
        from graph inputs, constants and sizes with the ops in _POSITION_OPS
        only, or an empty set otherwise.

        Integers are exact, so they only need to move with the angle when they
        read nothing else. Then the casts I64toI32 puts after int64 producers
        move together with those producers.
        """
        # An explicit stack, since an integer chain can be longer than the
        # recursion limit.
        stack = [node]
        while stack:
            top = stack[-1]
            if top in positions:
                stack.pop()
                continue
            args = [
                arg
                for arg in top.all_input_nodes
                if arg.op == "call_function"
                and not isinstance(arg.meta.get("val"), torch.SymInt)
            ]
            pending = [arg for arg in args if _is_exact(arg) and arg not in positions]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            nodes = {top}
            for arg in args:
                arg_nodes = positions[arg] if _is_exact(arg) else set()
                if not arg_nodes:
                    nodes = set()
                    break
                nodes |= arg_nodes
            if top.target not in _POSITION_OPS or QCOM_QUANT_ATTRS in top.meta:
                nodes = set()
            positions[top] = nodes
        return positions[node]
