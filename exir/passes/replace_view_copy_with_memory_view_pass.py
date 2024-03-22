# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from executorch.exir import memory
from executorch.exir.dialects._ops import ops
from executorch.exir.memory_planning import is_op_memory_planned
from executorch.exir.tensor import (
    contiguous_stride_from_shape,
    determine_tensor_dynanism,
    dim_order_from_stride,
    TensorShapeDynamism,
    TensorSpec,
)
from torch.export.exported_program import ExportedProgram, ExportGraphSignature
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)


def _is_view_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.view_copy.default,
        ops.edge.aten.view_copy.default,
    )


class _MemoryViewSpec(TensorSpec):
    def __init__(self, base: TensorSpec, shape: List[int]) -> None:
        """
        A MemoryViewSpec is an immutable TensorSpec that mirrors its base for non-size
        related information.
        """

        if math.prod(base.shape) != math.prod(shape):
            raise Exception(
                f"Cannot create a MemoryViewSpec because the provided shape {shape} is not consistent with the number of elements in the provided base ({math.prod(base.shape)})."
            )

        if not base.is_static_shape_tensor:
            raise Exception(
                "Cannot create a MemoryViewSpec because the provided base is not a static shape tensor."
            )

        self._init_setters = [
            "_frozen",
            "_base",
            "_guards",
            "shape",
            "stride",
            "dim_order",
            "shape_dynamism",
        ]
        self._frozen = False
        self._base = base
        self.shape: List[int] = shape
        self.stride: Tuple[int] = contiguous_stride_from_shape(torch.Size(self.shape))
        self.dim_order: Tuple[bytes] = dim_order_from_stride(self.stride)
        self.shape_dynamism: TensorShapeDynamism = determine_tensor_dynanism(
            torch.Size(self.shape)
        )

        # This spec gives a view into its base.
        # The base can be modified (e.g., mem_id) and this spec will
        # update accordingly, but certain fields we do not expect to change
        # We create guards for these
        self._guards: Dict[str, Any] = {
            "shape_dynamism": base.shape_dynamism,
            "scalar_type": base.scalar_type,
            "layout": base.layout,
            "is_sparse": base.is_sparse,
        }
        self._frozen = True

    def _check_guards(self) -> None:
        for name in self._guards:
            if getattr(self._base, name) != self._guards[name]:
                raise Exception(
                    f"The guarded attribute '{name}' has changed value.  At creation of the MemoryViewSpec, it was {self._guards[name]}, but it is now {getattr(self._base, name)}."
                )

    def __getattribute__(self, name):  # pyre-ignore
        if name in [
            "_init_setters",
            "_frozen",
            "_base",
            "_guards",
            "_check_guards",
            # Adding debug is needed so that view_spec.debug() shows the right id in
            # its string (if debug is excluded, it shows the id(view_spec._base) instead
            # of id(view_spec))
            "debug",
        ]:
            return object.__getattribute__(self, name)

        # Guard check after freeze
        if self._frozen:
            self._check_guards()

        # self._init_setters attributes come from self, others come from base
        if name in self._init_setters:
            return object.__getattribute__(self, name)
        return getattr(self._base, name)

    def __setattr__(self, name: str, val) -> None:  # pyre-ignore
        if name in ["_init_setters", "_frozen"]:
            object.__setattr__(self, name, val)
            return

        # Allow setting during initialization
        if name in self._init_setters and not self._frozen:
            object.__setattr__(self, name, val)
            return

        if name in self._init_setters:
            raise Exception(
                f"MemoryViewSpec is immutable.  Cannot set the attribute '{name}' after creation."
            )

        raise Exception(
            f"MemoryViewSpec is immutable.  To update the non-size related attribute '{name}', update the base."
        )


class ReplaceViewCopyWithMemoryViewPass(PassBase):
    def __init__(self) -> None:
        super().__init__()
        self._graph_signature: Optional[ExportGraphSignature] = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """
        This pass replaces all static, memory-planned view_copy nodes with special
        memory.view nodes.

        This should be run after the NormalizeViewCopyBasePass.

        During memory planning, memory.view nodes share the same storage as their base.

        During emission, memory.view nodes are not emitted as operators, but are instead
        directly emitted as evalues.  They share the same allocation_info/storage as their
        base.
        """

        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if self._is_replaceable_view_copy(node):
                    base, _ = node.args
                    node.target = memory.view

                    # Create spec for the node.
                    # _MemoryViewSpec is an immutable TensorSpec gives a view into
                    # its base spec for non-size related information.

                    # the shape is not the same as node.args[1] because node.args[1]
                    # can have an inferred sizes (-1).  This is also better if we want to
                    # extend support of memory.view for dynamic shapes in the future (which requires
                    # and operator that implements memory.view).
                    shape = node.meta["val"].shape
                    node.meta["spec"] = _MemoryViewSpec(base.meta["spec"], shape)

                    n_replaced += 1

            module.recompile()

        logger.debug(f"Replaced {n_replaced} view_copy nodes with memory.view nodes.")
        return PassResult(graph_module, n_replaced > 0)

    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                assert not self._is_replaceable_view_copy(node)
                if node.op == "call_function" and node.target == memory.view:
                    assert isinstance(node.meta["spec"], _MemoryViewSpec)

    def requires(self, graph_module: torch.fx.GraphModule) -> None:
        """
        This pass should be called after NormalizeViewCopyBasePass.
        We check that all view_copy nodes have been normalized.
        """
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if _is_view_copy(node):
                    base, size = node.args
                    assert not _is_view_copy(base)

    def set_graph_signature(self, graph_signature: ExportGraphSignature) -> None:
        self._graph_signature = graph_signature

    def _is_replaceable_view_copy(self, node: torch.fx.Node) -> bool:
        if not _is_view_copy(node):
            return False

        base = node.args[0]
        assert isinstance(base, torch.fx.Node)

        is_base_memory_planned = False  # until proven otherwise
        if base.op == "call_function":
            is_base_memory_planned = is_op_memory_planned(base)

        if base.op == "placeholder":
            # For now, we only assume placeholder parameters + buffers are memory planned.
            # We cautiously assume that general user inputs are not memory planned.
            # Memory planning for user inputs can be turned off, see
            # https://github.com/pytorch/executorch/blob/main/exir/passes/memory_planning_pass.py#L32

            if self._graph_signature is None or not isinstance(
                self._graph_signature, ExportGraphSignature
            ):
                logger.warning(
                    "The ExportGraphSignature was not set prior to calling ReplaceViewCopyWithMemoryViewPass.  Placeholder view_copy nodes cannot be replaced without first calling set_graph_signature."
                )
            else:
                if base.name in self._graph_signature.inputs_to_parameters:
                    is_base_memory_planned = True
                elif base.name in self._graph_signature.inputs_to_buffers:
                    is_base_memory_planned = True
                else:
                    pass

        return is_base_memory_planned and node.meta["spec"].is_static_shape_tensor
