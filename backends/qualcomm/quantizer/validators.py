# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import cast, Dict, List, Optional, Set, Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    QNN_QUANT_TYPE_MAP,
    QNN_TENSOR_TYPE_MAP,
)
from executorch.backends.qualcomm.utils.constants import QCOM_BLOCK_SIZE
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


@dataclass
class TensorQuantConstraint:
    """
    Normalized view of QuantConstraintInfo for a single tensor port.
    """

    encoding_types: List[PyQnnManager.Qnn_QuantizationEncoding_t] = field(
        default_factory=list
    )
    is_symmetric: bool = False
    axis: Optional[List[int]] = None  # per-channel axis list; None for per-tensor
    is_math_invariant: bool = False
    scale: Optional[float] = None
    offset: Optional[int] = None


@dataclass
class PortDatatypeConstraints:
    """
    Normalized view for a single input/output port:
      - selected dtype
      - all quant constraints available for that dtype
    """

    dtype: PyQnnManager.Qnn_DataType_t = (
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UNDEFINED
    )
    constraints: List[TensorQuantConstraint] = field(default_factory=list)
    applicable_from_current_dtype_onward: bool = False


@dataclass
class NormalizedConstraints:
    """
    Normalized constraints for an OpInfo variant:
      - inputs: per-port normalized constraints
      - outputs: per-port normalized constraints
    """

    inputs: List[PortDatatypeConstraints] = field(default_factory=list)
    outputs: List[PortDatatypeConstraints] = field(default_factory=list)


class ConstraintCache:
    def __init__(self):
        self._store: Dict[str, List[NormalizedConstraints]] = {}

    def get(self, key: str) -> Optional[NormalizedConstraints]:
        return self._store.get(key)

    def put(self, key: str, value: List[NormalizedConstraints]):
        self._store[key] = value


def _get_qnn_datatype(node: Node, qspec: QuantizationSpecBase):
    # TODO: Support multi-output use case
    meta_val = (
        node.meta["val"][0]
        if isinstance(node.meta["val"], (list, tuple))
        else node.meta["val"]
    )
    torch_dtype = meta_val.dtype

    if qspec:
        quant_max = qspec.quant_max
        quant_min = qspec.quant_min
        torch_dtype = qspec.dtype
        # quant_max and quant_min are optional, so we need to handle the case where they are None
        if quant_max is not None and quant_min is not None:
            quant_range = quant_max - quant_min
            unsigned = quant_min >= 0
            if quant_range <= torch.iinfo(torch.int8).max - torch.iinfo(torch.int8).min:
                torch_dtype = torch.uint8 if unsigned else torch.int8

            elif (
                quant_range
                <= torch.iinfo(torch.int16).max - torch.iinfo(torch.int16).min
            ):
                torch_dtype = torch.uint16 if unsigned else torch.int16

    return QNN_QUANT_TYPE_MAP.get(torch_dtype, QNN_TENSOR_TYPE_MAP[torch_dtype])


def _resolve_shared_qspec(
    spec: Optional[QuantizationSpecBase],
    visited: Optional[Set[torch.fx.Node]] = None,
) -> Optional[QuantizationSpecBase]:
    """
    Resolve a SharedQuantizationSpec into a concrete QuantizationSpecBase.

    Resolution strategy:
      1) If `spec` is not a SharedQuantizationSpec, return it directly.
      2) If it *is* a SharedQuantizationSpec, use `spec.edge_or_node` to locate
         the source node where the concrete qspec originates.
      3) If second_node exists, choose to the
        `second_node`'s `input_qspec_map[source_node]`, then resolve recursively.
      4) Using the source node's `annotation.output_qspec`. If that is
         also shared, resolve recursively.
      5) If nothing can be resolved, return None.

    A `visited` set is used to avoid infinite recursion when malformed
    annotations or cyclic references exist.

    Args:
        spec:         The qspec to be resolved (SharedQuantizationSpec).
        visited:      Internal recursion guard to detect cycles (optional).

    Returns:
        A concrete QuantizationSpec if successfully resolved; otherwise None.
    """
    if spec is None:
        return None

    if not isinstance(spec, SharedQuantizationSpec):
        # Already concrete; nothing to resolve.
        return spec

    visited = visited or set()

    # `edge_or_node` may be a (node, edge) tuple or a node directly.
    edge_or_node = spec.edge_or_node
    second_node = None
    if isinstance(edge_or_node, tuple):
        source_node, second_node = edge_or_node
    else:
        source_node = edge_or_node

    # Cycle guard to avoid infinite recursion.
    if source_node in visited:
        return None
    visited.add(source_node)

    # First, try to look up the spec from the current node's input map.
    # This is because in the case of mix precision, the second node
    # might have a specific quantization spec for the input.
    if second_node:
        sec_ann = second_node.meta.get(Q_ANNOTATION_KEY, None)
        if sec_ann and (in_spec := sec_ann.input_qspec_map.get(source_node, None)):
            return _resolve_shared_qspec(in_spec, visited)

    # use the source node's own output qspec.
    src_ann = source_node.meta.get(Q_ANNOTATION_KEY, None)
    if src_ann and (out_spec := src_ann.output_qspec):
        return _resolve_shared_qspec(out_spec, visited)

    # Nothing found.
    return None


def _get_output_qspec_from_node(
    quantization_annotation: QuantizationAnnotation,
) -> QuantizationSpecBase | None:
    out_spec = quantization_annotation.output_qspec
    if isinstance(out_spec, SharedQuantizationSpec):
        out_spec = _resolve_shared_qspec(out_spec)
    return out_spec


def _get_input_qspecs_from_node(
    quantization_annotation: QuantizationAnnotation,
) -> Dict[Node, QuantizationSpecBase | None]:
    input_qspec_map = quantization_annotation.input_qspec_map
    found_qspecs = {}

    for node, spec in input_qspec_map.items():
        resolved_spec = spec
        if isinstance(spec, SharedQuantizationSpec):
            resolved_spec = _resolve_shared_qspec(resolved_spec)
        found_qspecs[node] = resolved_spec

    return found_qspecs


def _node_port_dtypes(node) -> Tuple[List[str], List[str]]:
    """
    Extract input/output dtypes from FX node meta.
    Adjust this function to your actual FX meta layout.
    """
    in_dtypes, out_dtypes = [], []
    if Q_ANNOTATION_KEY not in node.meta:
        return in_dtypes, out_dtypes
    quantization_annotation = cast(
        QuantizationAnnotation, node.meta.get(Q_ANNOTATION_KEY)
    )
    input_qspecs = _get_input_qspecs_from_node(quantization_annotation)
    output_qspec = _get_output_qspec_from_node(quantization_annotation)

    for arg in node.args:
        if isinstance(arg, Node) and "val" in arg.meta:
            in_dtypes.append(_get_qnn_datatype(arg, input_qspecs.get(arg, None)))
        elif isinstance(arg, (tuple, list)):  # for concat op
            for item in arg:
                if isinstance(item, Node) and "val" in item.meta:
                    in_dtypes.append(
                        _get_qnn_datatype(item, input_qspecs.get(item, None))
                    )

    # If the output is not annotated, we validate only the inputs, such as aten.max.dim
    # TODO: Support multi-output use case
    if output_qspec is not None:
        out_dtypes.append(_get_qnn_datatype(node, output_qspec))

    return in_dtypes, out_dtypes


def _is_nc_compatible_with_node(
    normalized_constraints: NormalizedConstraints,
    in_dtypes: List[str],
    out_dtypes: List[str],
) -> bool:
    """
    Basic dtype compatibility check between FX node and an NormalizedConstraints variant.
    """
    nc_inputs = (
        normalized_constraints.inputs * len(in_dtypes)
        if getattr(
            normalized_constraints.inputs[0],
            "applicable_from_current_dtype_onward",
            False,
        )
        else normalized_constraints.inputs
    )
    nc_outputs = (
        normalized_constraints.outputs * len(out_dtypes)
        if getattr(
            normalized_constraints.outputs[0],
            "applicable_from_current_dtype_onward",
            False,
        )
        else normalized_constraints.outputs
    )

    # It is acceptable for the number of constraints to be greater than the number of inputs and outputs.
    if len(in_dtypes) > len(nc_inputs) or len(out_dtypes) > len(nc_outputs):
        return False

    for i, dt in enumerate(in_dtypes):
        if dt != nc_inputs[i].dtype:
            return False

    for i, dt in enumerate(out_dtypes):
        if dt != nc_outputs[i].dtype:
            return False

    return True


def _check_symmetric(qspec: QuantizationSpecBase):
    return qspec.qscheme in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]


def _qspec_port_encoding_type(node: Node, qspec: QuantizationSpecBase):
    encoding_type = None
    qscheme = qspec.qscheme

    if qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        if qspec.dtype == torch.int8 and qspec.quant_max - qspec.quant_min <= 15:
            encoding_type = (
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET
            )
        else:
            encoding_type = (
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            )
    elif qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        if (
            obs_or_fq := getattr(qspec, "observer_or_fake_quant_ctr", None)
        ) and obs_or_fq.p.keywords.get(QCOM_BLOCK_SIZE, None) is not None:
            encoding_type = (
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
            )
        elif qspec.dtype == torch.int8 and qspec.quant_max - qspec.quant_min <= 15:
            encoding_type = (
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
            )
        else:
            encoding_type = (
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
            )
    else:
        logging.warning(f"Node ({node.name}) failed to get encoding type")
    return encoding_type


def _check_math_invariant(qspec: QuantizationSpecBase) -> bool:
    return isinstance(qspec, SharedQuantizationSpec)


def _check_scale_and_offset(
    qspec: QuantizationSpecBase, scale: float, offset: int
) -> bool:
    valid = True
    obs_or_fqc = qspec.observer_or_fake_quant_ctr()
    if getattr(obs_or_fqc, "scale", None) is not None:
        valid &= obs_or_fqc.scale == torch.tensor(scale)
    if getattr(obs_or_fqc, "zero_point", None) is not None:
        valid &= obs_or_fqc.zero_point == -offset
    return valid


def validate_against_backend_constraints(  # noqa: C901
    node: Node, normalized_constraints_list: List[NormalizedConstraints]
) -> bool:
    valid = True

    quantization_annotation = cast(
        QuantizationAnnotation, node.meta.get(Q_ANNOTATION_KEY)
    )
    in_dtypes, out_dtypes = _node_port_dtypes(node)
    # Cannot validate without proper dtypes.
    if len(in_dtypes) == 0 and len(out_dtypes) == 0:
        logging.warning(
            f"Skipping validation for Node ({node.name}) because it does not have the appropriate data types. quantization_annotation: {quantization_annotation}"
        )
        return valid

    nc = None
    for normalized_constraints in normalized_constraints_list:
        if _is_nc_compatible_with_node(normalized_constraints, in_dtypes, out_dtypes):
            nc = normalized_constraints
            break

    if nc is None:
        # For custom operators, since we don't have operator information, we're unable to validate the constraints.
        # We assume they are valid, and validation should be handled by the backend.
        logging.warning(
            f"Skipping validation for Node ({node.name}) as a compatible dtype was not found in op_info. quantization_annotation: {quantization_annotation}"
        )
        return valid

    input_qspec_map = quantization_annotation.input_qspec_map
    if input_qspec_map is not None:
        for (input_node, input_qspec), nc_input in zip(
            quantization_annotation.input_qspec_map.items(), nc.inputs
        ):
            for quant_constraint in nc_input.constraints:
                resolved_input_qspec = _resolve_shared_qspec(input_qspec)
                encoding_type = _qspec_port_encoding_type(
                    input_node, resolved_input_qspec
                )
                if encoding_type in quant_constraint.encoding_types:
                    if quant_constraint.is_symmetric and not _check_symmetric(
                        resolved_input_qspec
                    ):
                        logging.warning(
                            f"Input node ({input_node.name}) of node ({node.name}) failed to meet symmetric constraint"
                        )
                        valid = False

                    if (
                        quant_constraint.is_math_invariant
                        and not _check_math_invariant(input_qspec)
                    ):
                        logging.warning(
                            f"Input node ({input_node.name}) of node ({node.name}) failed to meet math invariant constraint"
                        )
                        valid = False
                    if (
                        quant_constraint.scale is not None
                        and quant_constraint.offset is not None
                    ) and not _check_scale_and_offset(
                        resolved_input_qspec,
                        quant_constraint.scale,
                        quant_constraint.offset,
                    ):
                        logging.warning(
                            f"Input node ({input_node.name}) of node ({node.name}) failed to meet scale ({quant_constraint.scale}) and offset ({quant_constraint.offset}) constraint"
                        )
                        valid = False
                    break

            if not valid:
                logging.warning(
                    f"Node ({input_node.name})'s input_qspec {input_qspec} failed to match constraints {quant_constraint}"
                )
                return valid

    output_qspec = quantization_annotation.output_qspec
    if output_qspec is not None:
        resolved_output_qspec = _resolve_shared_qspec(output_qspec)
        # TODO: Support multi-output use case
        for quant_constraint in nc.outputs[0].constraints:
            encoding_type = _qspec_port_encoding_type(node, resolved_output_qspec)
            if encoding_type in quant_constraint.encoding_types:
                if quant_constraint.is_symmetric is not None and not _check_symmetric(
                    resolved_output_qspec
                ):
                    logging.warning(
                        f"Node ({node.name}) failed to meet symmetric constraint"
                    )
                    valid = False
                if (
                    quant_constraint.is_math_invariant is not None
                    and not _check_math_invariant(output_qspec)
                ):
                    logging.warning(
                        f"Node ({node.name}) failed to meet math invariant constraint"
                    )
                    valid = False
                if (
                    quant_constraint.scale is not None
                    and quant_constraint.offset is not None
                ) and not _check_scale_and_offset(
                    resolved_output_qspec,
                    quant_constraint.scale,
                    quant_constraint.offset,
                ):
                    logging.warning(
                        f"Node ({node.name}) failed to meet scale ({quant_constraint.scale}) and offset ({quant_constraint.offset}) constraint"
                    )
                    valid = False

        if not valid:
            logging.warning(
                f"Node ({node.name})'s output_qspec {output_qspec} failed to match constraints {quant_constraint}"
            )

    return valid
