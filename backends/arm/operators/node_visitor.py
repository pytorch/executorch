# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide utilities to register and apply TOSA node visitors.

Use this module to construct and serialize TOSA operators from FX nodes.
- Define the NodeVisitor base class and registry
- Register concrete visitors per TOSA specification

"""

import json

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import tosa_serializer as ts

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.debug.schema import DebugHook
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import (
    TosaSpecification,
    TosaSpecMapping,
)
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)


class NodeVisitor:
    """Provide a visitor pattern to lower edge IR to TOSA.

    Attributes:
        _exported_program (torch.export.ExportedProgram): Source program being lowered.
        tosa_spec (TosaSpecification): Active TOSA specification for lowering.
        debug_hook (Optional[DebugHook]): Optional hook for debug metadata.

    """

    # Add the currently supported node_visitor specs as default.
    # This should be overriden in the NodeVisitor subclasses to target
    # a specific TOSA version.
    # When all node_visitors has been refactored to target a specific
    # version, this list should be removed.
    tosa_specs = TosaSpecification.all_versions_and_profiles()

    def __init__(
        self,
        exported_program: ExportedProgram,
        tosa_spec: TosaSpecification,
        debug_hook: Optional[DebugHook] = None,
    ):
        self._exported_program = exported_program
        self.tosa_spec = tosa_spec
        self.debug_hook = debug_hook

    def _serialize_operator(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        tosa_op: ts.Op,
        inputs: List[str],
        outputs: List[str],
        attributes: Optional[Any] = None,
    ) -> None:
        """Serialize a TOSA operator into the graph.

        When a ``DebugHook`` is active, attach location metadata (in JSON) to
        the operator for traceability.

        Args:
            node (torch.fx.Node): Source FX node being lowered.
            tosa_graph: Target TOSA serializer/graph object.
            tosa_op: TOSA operator enum value to emit.
            inputs (List[str]): Names of input tensors.
            outputs (List[str]): Names of output tensors.
            attributes (Optional[Any]): Optional TOSA attribute object.

        Returns:
            None: Mutates ``tosa_graph`` in place.

        """
        op_location = None
        if self.debug_hook:
            debug_info = self.debug_hook.add(
                node,
                tosa_op=outputs[0],
                tosa_op_id=tosa_op,
            )

            if self.debug_hook.mode == ArmCompileSpec.DebugMode.TOSA:
                op_location = json.dumps(debug_info.to_dict())

        tosa_graph.addOperator(
            tosa_op,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes,
            location=op_location,
        )

    def validate(
        self,
        *,
        target: str,
        inputs: List[TosaArg],
        output: TosaArg,
        num_inputs: int | List[int],
        input_dtypes: List[Any],
        output_dtypes: Optional[List[Any]] = None,
        same_dtype_with_output: bool = True,
        dtype_check_inputs_only: bool = False,
    ) -> None:
        validate_num_inputs(target, inputs, num_inputs)
        if same_dtype_with_output:
            validate_same_dtype(target, [*inputs, output], ts)
        else:
            validate_same_dtype(target, inputs, ts)

        dtype_check_tensors = inputs if dtype_check_inputs_only else [*inputs, output]
        validate_valid_dtype(
            target,
            dtype_check_tensors,
            input_dtypes,
            self.tosa_spec,
        )
        if output_dtypes is not None:
            validate_valid_dtype(
                target,
                output,
                output_dtypes,
                self.tosa_spec,
            )

    def serialize(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        *,
        tosa_op: ts.Op,
        inputs: List[TosaArg],
        output: TosaArg,
        attr_method: Optional[str] = None,
        attr_kwargs: Optional[dict[str, Any]] = None,
        attr_builder: Optional[Callable[[ts.TosaSerializerAttribute], None]] = None,
        extra_input_builders: Optional[
            List[Callable[[torch.fx.Node, Any, List[TosaArg], TosaArg, Any], str]]
        ] = None,
    ) -> None:
        attr = ts.TosaSerializerAttribute()
        if attr_method is not None:
            getattr(attr, attr_method)(**(attr_kwargs or {}))
        elif attr_builder is not None:
            attr_builder(attr)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define attr_method or attr_builder."
            )
        input_names = [arg.name for arg in inputs]
        for builder in extra_input_builders or []:
            input_names.append(
                builder(node, tosa_graph, inputs, output, self.tosa_spec)
            )
        self._serialize_operator(
            node,
            tosa_graph,
            tosa_op,
            input_names,
            [output.name],
            attr,
        )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        """Define a TOSA operator node.

        Args:
            node (torch.fx.Node): FX node being lowered.
            tosa_graph (serializer.tosa_serializer.TosaSerializer): Target TOSA graph.
            inputs (List[TosaArg]): Input tensor arguments.
            output (TosaArg): Output tensor descriptor.

        Returns:
            None: Mutates ``tosa_graph`` in place.

        Raises:
            ValueError: If input count or dtypes are invalid.

        """
        raise NotImplementedError("NodeVisitor must be extended.")


# container for all node visitors
_node_visitor_tuples: TosaSpecMapping[tuple] = TosaSpecMapping()


def register_node_visitor(visitor):
    """Register a concrete ``NodeVisitor`` class for its TOSA specs."""
    for tosa_spec in visitor.tosa_specs:
        # Try to get the tuple to make sure it doesn't exist
        visitor_tuple = (visitor.target, visitor)
        try:
            tuples = _node_visitor_tuples.get(tosa_spec)
        except KeyError:
            tuples = []

        if visitor_tuple in tuples:
            raise RuntimeError(
                f"Visitor for target {visitor.target} already registered for TOSA spec {tosa_spec}"
            )
        _node_visitor_tuples.add(tosa_spec, visitor_tuple)
    return visitor


def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    """Return a mapping from target names to visitor instances for a spec."""
    # Ensure all operator modules are imported so visitors are registered.
    import executorch.backends.arm.operators  # noqa: F401

    node_visitors: Dict[str, NodeVisitor] = {}
    tosa_spec: TosaSpecification | None = None
    for arg in args:
        if isinstance(arg, TosaSpecification):
            tosa_spec = arg
            break

    if tosa_spec is None:
        raise RuntimeError("No TOSA specification supplied.")

    # Use the mapping to get the dict for this spec (handles combined specs)
    for node_visitor_tuple in _node_visitor_tuples.get(tosa_spec):
        target, visitor = node_visitor_tuple
        if target in node_visitors and node_visitors[target].__class__ != visitor:
            logger.warning(
                f"Target {target} already has visitor class {node_visitors[target].__class__.__name__} registered, overwriting with class: {visitor.__name__}"
            )
        node_visitors[target] = visitor(*args)

    return node_visitors
