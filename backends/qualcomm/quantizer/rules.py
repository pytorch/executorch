# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from executorch.backends.qualcomm.quantizer.validators import (
    NormalizedConstraints,
    validate_against_backend_constraints,
)
from executorch.backends.qualcomm.serialization.qc_schema import SocInfo

from executorch.backends.qualcomm.utils.constants import QCOM_BLOCK_SIZE
from torch._ops import OpOverload
from torch._subclasses import FakeTensor
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from .qconfig import QuantizationConfig


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if Q_ANNOTATION_KEY not in node.meta:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
        node.meta[Q_ANNOTATION_KEY]._annotated = True


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            Q_ANNOTATION_KEY in node.meta and node.meta[Q_ANNOTATION_KEY]._annotated
        )
    return annotated


def _is_float_tensor(node: Node):
    """Check if the node's tensor is a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if (
        not isinstance(node, Node)
        or "val" not in node.meta
        or not isinstance(node.meta["val"], FakeTensor)
    ):
        return False
    return node.meta["val"].dtype in (torch.bfloat16, torch.float32)


def annotate_in_out_obs_sharing_op(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    if _is_annotated([node]):
        return

    input_act = node.args[0]
    assert isinstance(input_act, Node)

    # only annotate input output sharing operator
    # when the output of the input node is annotated
    if (
        Q_ANNOTATION_KEY not in input_act.meta
        or not input_act.meta[Q_ANNOTATION_KEY]._annotated
        or input_act.meta[Q_ANNOTATION_KEY].output_qspec is None
        or not _is_float_tensor(input_act)
    ):
        return

    act_qspec = SharedQuantizationSpec(input_act)
    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map={
            input_act: act_qspec,
        },
        output_qspec=act_qspec,
        _annotated=True,
    )


def annotate_single_in_share_out(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act_qspec = quantization_config.input_activation
    input_act = node.args[0]
    if _is_float_tensor(input_act) and input_act_qspec is not None:
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

    output_act_qspec = (
        SharedQuantizationSpec((input_act, node))
        if _is_float_tensor(node) and input_act_qspec is not None
        else None
    )
    if len(input_qspec_map) > 0 or output_act_qspec is not None:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


def annotate_single_in(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act_qspec = quantization_config.input_activation
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    if input_act_qspec is not None:
        input_qspec_map[input_act] = input_act_qspec

    if len(input_qspec_map) > 0:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )


def annotate_single_in_single_out(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act_qspec = quantization_config.input_activation
    if _is_float_tensor(node.args[0]) and input_act_qspec is not None:
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

    output_act_qspec = (
        quantization_config.output_activation if _is_float_tensor(node) else None
    )

    if len(input_qspec_map) > 0 or output_act_qspec is not None:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


def annotate_binary(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = (
        quantization_config.output_activation if _is_float_tensor(node) else None
    )

    input_qspec_map = {}
    input_act0 = node.args[0]
    if _is_float_tensor(input_act0) and input_act_qspec is not None:
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if _is_float_tensor(input_act1) and input_act_qspec is not None:
        input_qspec_map[input_act1] = input_act_qspec

    if len(input_qspec_map) > 0 or output_act_qspec is not None:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


def annotate_conv(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    # block quantization
    if quantization_config.block_size is not None:
        quantization_config.weight.observer_or_fake_quant_ctr.p.keywords.update(
            {QCOM_BLOCK_SIZE: quantization_config.block_size}
        )

    input_qspec_map = {}
    input_act_qspec = quantization_config.input_activation
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    if input_act_qspec is not None:
        input_qspec_map[input_act] = input_act_qspec

    weight = node.args[1]
    assert isinstance(weight, Node)
    input_qspec_map[weight] = quantization_config.weight

    if len(node.args) > 2:
        bias = node.args[2]
        if isinstance(bias, Node):
            if callable(quantization_config.bias):
                input_qspec_map[bias] = quantization_config.bias(node)
            else:
                input_qspec_map[bias] = quantization_config.bias

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )


@dataclass(frozen=True)
class OpQuantRule:
    """
    A single rule encapsulating:
      - ATen → QNN op mapping
      - backend-specific annotation (annotate_fn)
      - validation via unified constraints
    """

    # --- Mapping (ATen → QNN OP) ---
    aten_op: OpOverload  # e.g., "aten.conv2d"
    qnn_op: Optional[str]  # None → no mapping → skip validation

    # --- Annotation (backend-specific) ---
    annotate_fn: Optional[Callable[[Node, QuantizationConfig], None]]

    # --- Validation (backend-specific constraints) ---
    validate_fn: Optional[Callable[[Node, NormalizedConstraints, SocInfo], bool]] = (
        None  # Custom validation logic
    )


class GeneralOpDef:
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig):
        annotate_single_in_single_out(node, quantization_config)

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        # If there's no quantization annotation, we can't validate against constraints.
        if not _is_annotated([node]):
            return True
        return validate_against_backend_constraints(node, constraints_list)
