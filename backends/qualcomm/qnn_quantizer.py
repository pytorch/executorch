# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn.functional as F

from executorch.backends.qualcomm.passes.convert_hardsigmoid import ConvertHardsigmoid
from executorch.backends.qualcomm.passes.reduce_dynamic_range import ReduceDynamicRange
from executorch.backends.qualcomm.passes.remove_clone import RemoveClone

from torch import Tensor
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.quantize_pt2e import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.ao.quantization.quantizer import SharedQuantizationSpec
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    _is_sym_size_node,
    _node_only_used_for_sym_size,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

__all__ = [
    "QnnQuantizer",
    "get_default_qnn_ptq_config",
    "get_16bit_qnn_ptq_config",
]

QUANT_ANNOTATION_KEY = "quantization_annotation"


@dataclass(repr=False, eq=False, frozen=True)
class QnnQuantizerConfig:
    enable_per_channel_conv_quant: bool


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec | Callable]


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            QUANT_ANNOTATION_KEY in node.meta
            and node.meta[QUANT_ANNOTATION_KEY]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if QUANT_ANNOTATION_KEY not in node.meta:
                node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation()
            node.meta[QUANT_ANNOTATION_KEY]._annotated = True


def _derived_bias_quant_spec(node: Node) -> DerivedQuantizationSpec:
    def _derive_bias_qparams_fn(
        obs_or_fqs: List,
    ) -> Tuple[Tensor, Tensor]:
        assert (
            len(obs_or_fqs) == 2
        ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        act_obs_or_fq = obs_or_fqs[0]
        weight_obs_or_fq = obs_or_fqs[1]
        weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
        act_scale, act_zp = act_obs_or_fq.calculate_qparams()
        (broadcast_act_scale, broadcast_weight_scale) = torch.broadcast_tensors(
            act_scale, weight_scale
        )
        derived_scale = (broadcast_act_scale * broadcast_weight_scale).to(torch.float32)
        derived_zero = torch.zeros(derived_scale.size()).to(torch.int32)
        return (derived_scale, derived_zero)

    input_act = node.args[0]
    assert isinstance(input_act, Node)
    weight = node.args[1]
    assert isinstance(weight, Node)

    return DerivedQuantizationSpec(
        derived_from=[(input_act, node), (weight, node)],
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        ch_axis=0,
        qscheme=torch.per_channel_symmetric,
    )


def get_default_qnn_ptq_config(
    enable_per_channel_conv_quant=False,
) -> Tuple[QuantizationConfig, QnnQuantizerConfig]:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    quantizer_configs = QnnQuantizerConfig(
        enable_per_channel_conv_quant=enable_per_channel_conv_quant
    )

    return quantization_config, quantizer_configs


def get_16bit_qnn_ptq_config() -> Tuple[QuantizationConfig, QnnQuantizerConfig]:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=0,
        quant_max=65535,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=-32767,
        quant_max=32767,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    quantizer_configs = QnnQuantizerConfig(enable_per_channel_conv_quant=False)

    return quantization_config, quantizer_configs


def get_ptq_per_channel_weight_config() -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = _derived_bias_quant_spec

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )
    return quantization_config


class QnnQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.configs: Optional[QnnQuantizerConfig] = None
        self.global_quant_config: Optional[QuantizationConfig] = None
        self.custom_quant_configs: Dict[
            Union[Type[torch.nn.Module], Callable], Optional[QuantizationConfig]
        ] = {}
        self.custom_quant_annotations: Tuple[Callable] = ()

    def set_global_op_quant_config(
        self, quantization_config: Tuple[QuantizationConfig, QnnQuantizerConfig]
    ) -> None:
        quant_config, quantizer_config = quantization_config
        self.global_quant_config = quant_config
        self.configs = quantizer_config

    def add_custom_quant_annotations(
        self, custom_quant_annotations: Tuple[Callable]
    ) -> None:
        self.custom_quant_annotations = custom_quant_annotations

    def _get_quant_config(
        self, ops: List[Union[Type[torch.nn.Module], Callable]]
    ) -> Optional[QuantizationConfig]:
        """
        Priority:
            1. per channel config when enable_per_channel_conv_quant is True
            2. global config
        """

        conv_ops = {torch.nn.Conv2d, torch.nn.functional.conv2d}
        if self.configs.enable_per_channel_conv_quant and set(ops).intersection(
            conv_ops
        ):
            return get_ptq_per_channel_weight_config()

        return self.global_quant_config

    def _annotate_input_out_obs_sharing_op(
        self,
        ops: List[Union[Type[torch.nn.Module], Callable]],
        gm: torch.fx.GraphModule,
    ) -> None:
        module_partitions = get_source_partitions(gm.graph, ops)
        partitions = list(itertools.chain(*module_partitions.values()))
        for partition in partitions:
            io_obs_sharing_node = partition.output_nodes[0]
            if _is_annotated([io_obs_sharing_node]):
                continue

            input_act = io_obs_sharing_node.args[0]
            assert isinstance(input_act, Node)

            # only annotate input output sharing operator
            # when the output of the input node is annotated
            if (
                QUANT_ANNOTATION_KEY not in input_act.meta
                or not input_act.meta[QUANT_ANNOTATION_KEY]._annotated
                or input_act.meta[QUANT_ANNOTATION_KEY].output_qspec is None
            ):
                continue

            act_qspec = SharedQuantizationSpec(input_act)
            io_obs_sharing_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map={
                    input_act: act_qspec,
                },
                output_qspec=act_qspec,
                _annotated=True,
            )

    def _annotate_binary(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
            "add",
            "sub",
            "mul",
            "div",
            "truediv",
        ]

        quantization_config = self._get_quant_config(op_sources)
        binary_partitions = get_source_partitions(gm.graph, op_sources)
        binary_partitions = list(itertools.chain(*binary_partitions.values()))
        for binary_partition in binary_partitions:
            node = binary_partition.output_nodes[0]
            if _is_annotated([node]):
                continue

            input_act_qspec = quantization_config.input_activation
            output_act_qspec = quantization_config.output_activation

            input_qspec_map = {}
            input_act0 = node.args[0]
            if isinstance(input_act0, Node):
                input_qspec_map[input_act0] = input_act_qspec

            input_act1 = node.args[1]
            if isinstance(input_act1, Node):
                input_qspec_map[input_act1] = input_act_qspec

            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )

    def _annotate_matmul(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [
            operator.matmul,
            torch.matmul,
            torch.ops.aten.matmul,
            "matmul",
        ]

        quantization_config = self._get_quant_config(op_sources)
        matmul_partitions = get_source_partitions(gm.graph, op_sources)
        matmul_partitions = list(itertools.chain(*matmul_partitions.values()))
        for matmul_partition in matmul_partitions:
            node = matmul_partition.output_nodes[0]
            if _is_annotated([node]):
                continue

            input_act_qspec = quantization_config.input_activation
            output_act_qspec = quantization_config.output_activation

            input_qspec_map = {}
            input_act0 = node.args[0]
            if isinstance(input_act0, Node):
                input_qspec_map[input_act0] = input_act_qspec

            input_act1 = node.args[1]
            if isinstance(input_act1, Node):
                # In matmul, QNN_DATATYPE_SFIXED_POINT_16 Input1 must have QNN_DATATYPE_UFIXED_POINT_16 Input0 and must be symmetric quantized.
                if input_act_qspec.dtype == torch.int32:
                    input_qspec_map[input_act1] = quantization_config.weight
                    quantization_annotation = input_act1.meta.get(
                        QUANT_ANNOTATION_KEY, None
                    )
                    if quantization_annotation:
                        quantization_annotation.output_qspec = (
                            quantization_config.weight
                        )
                else:
                    input_qspec_map[input_act1] = input_act_qspec

            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )

    def _annotate_relu(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.ReLU, F.relu]
        quantization_config = self._get_quant_config(op_sources)
        relu_partitions = get_source_partitions(gm.graph, op_sources)

        relu_partitions = list(itertools.chain(*relu_partitions.values()))
        for relu_partition in relu_partitions:
            if len(relu_partition.output_nodes) > 1:
                raise ValueError("Relu partition has more than one output node")
            relu_node = relu_partition.output_nodes[0]

            if not isinstance(relu_node, Node):
                raise ValueError(f"{relu_node} is not a Node")
            if relu_node.op != "call_function" or relu_node.target not in [
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
            ]:
                raise ValueError(f"{relu_node} is not an aten relu operator")

            if _is_annotated([relu_node]):
                continue

            input_qspec_map = {}
            input_act = relu_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = quantization_config.input_activation

            relu_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )

    def _annotate_hardswish(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.Hardswish]
        quantization_config = self._get_quant_config(op_sources)
        hardswish_partitions = get_source_partitions(gm.graph, op_sources)

        hardswish_partitions = list(itertools.chain(*hardswish_partitions.values()))
        for hardswish_partition in hardswish_partitions:
            if len(hardswish_partition.output_nodes) > 1:
                raise ValueError("Hardswish partition has more than one output node")
            hardswish_node = hardswish_partition.output_nodes[0]

            if not isinstance(hardswish_node, Node):
                raise ValueError(f"{hardswish_node} is not a Node")
            if hardswish_node.op != "call_function" or hardswish_node.target not in [
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ]:
                raise ValueError(f"{hardswish_node} is not an aten hardswish operator")

            if _is_annotated([hardswish_node]):
                continue

            input_qspec_map = {}
            input_act = hardswish_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = quantization_config.input_activation

            hardswish_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )

    def _annotate_hardsigmoid(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.Hardsigmoid]
        quantization_config = self._get_quant_config(op_sources)
        hardsigmoid_partitions = get_source_partitions(gm.graph, op_sources)

        hardsigmoid_partitions = list(itertools.chain(*hardsigmoid_partitions.values()))
        for hardsigmoid_partition in hardsigmoid_partitions:
            if len(hardsigmoid_partition.output_nodes) > 1:
                raise ValueError("Hardsigmoid partition has more than one output node")
            hardswish_node, div_node = hardsigmoid_partition.nodes

            if not isinstance(hardswish_node, Node):
                raise ValueError(f"{hardswish_node} is not a Node")
            if hardswish_node.op != "call_function" or hardswish_node.target not in [
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ]:
                raise ValueError(f"{hardswish_node} is not an aten hardswish operator")
            if div_node.op != "call_function" or div_node.target not in [
                torch.ops.aten.div.Tensor,
                torch.ops.aten.div_.Tensor,
            ]:
                raise ValueError(f"{div_node} is not an aten div operator")

            if _is_annotated([hardswish_node]):
                continue

            input_qspec_map = {}
            input_act = hardswish_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = quantization_config.input_activation

            hardswish_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )
            div_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )

    def _annotate_conv2d(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.Conv2d, torch.nn.functional.conv2d]
        quantization_config = self._get_quant_config(op_sources)
        conv_partitions = get_source_partitions(gm.graph, op_sources)
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue

            input_qspec_map = {}
            input_act = conv_node.args[0]
            assert isinstance(input_act, Node)
            input_spec = quantization_config.input_activation
            input_qspec_map[input_act] = input_spec

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = quantization_config.weight

            if len(conv_node.args) > 2:
                bias = conv_node.args[2]
                if isinstance(bias, Node):
                    if callable(quantization_config.bias):
                        input_qspec_map[bias] = quantization_config.bias(conv_node)
                    else:
                        input_qspec_map[bias] = quantization_config.bias

            conv_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )

    def _annotate_linear(self, gm: torch.fx.GraphModule) -> None:  # noqa: C901
        op_sources = [torch.nn.Linear, torch.nn.functional.linear]
        linear_partitions = get_source_partitions(gm.graph, op_sources)
        quantization_config = self._get_quant_config(op_sources)

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = quantization_config.output_activation
        weight_qspec = quantization_config.weight
        bias_qspec = quantization_config.bias
        linear_partitions = list(itertools.chain(*linear_partitions.values()))
        for linear_partition in linear_partitions:
            act_nodes = [
                n
                for n in linear_partition.input_nodes
                if not _node_only_used_for_sym_size(n, linear_partition.nodes)
            ]
            if len(act_nodes) > 1:
                raise ValueError(
                    f"Multiple activation nodes found for partition {linear_partition} {act_nodes}"
                )
            if len(act_nodes) == 0:
                raise ValueError(
                    f"No activation node found for partition {linear_partition}"
                )
            act_node = act_nodes[0]
            output_node = linear_partition.output_nodes[0]
            weight_node = None
            bias_node = None
            for node in linear_partition.params:
                weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                    weight_node = node
                if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                    bias_node = node
            if weight_node is None:
                raise ValueError("No weight found in Linear pattern")
            # find use of act node within the matched pattern
            act_use_node = None
            # When doing tracing with dynamic shape, we end up with sym_size nodes
            # This nodes do not need quantization, so skip those.
            # We can also have quant workflow throw exception when sym_size nodes
            # are annotated.
            # This is not specific to linear, so in future diffs we should streamline
            # this.
            act_node_users = list(
                filter((lambda x: (_is_sym_size_node(x) is False)), act_node.users)
            )
            act_use_node_in_p = set(act_node_users).intersection(
                set(linear_partition.nodes)
            )
            if len(act_use_node_in_p) != 1:
                raise ValueError(
                    f"Could not find a valid use of act node. All uses {act_use_node_in_p}"
                )
            act_use_node = act_use_node_in_p.pop()
            if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                _annotate_input_qspec_map(
                    act_use_node,
                    act_node,
                    input_act_qspec,
                )
            if bias_node and _is_annotated([bias_node]) is False:
                _annotate_input_qspec_map(act_use_node, bias_node, bias_qspec)
            if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                _annotate_input_qspec_map(act_use_node, weight_node, weight_qspec)
            if _is_annotated([output_node]) is False:
                _annotate_output_qspec(output_node, output_act_qspec)
            nodes_to_mark_annotated = list(linear_partition.nodes)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_hardtanh(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.modules.Hardtanh, torch.nn.modules.ReLU6]
        qconfig = self._get_quant_config(op_sources)
        hardtanh_partitions = get_source_partitions(gm.graph, op_sources)

        hardtanh_partitions = list(itertools.chain(*hardtanh_partitions.values()))
        for hardtanh_partition in hardtanh_partitions:
            if len(hardtanh_partition.output_nodes) > 1:
                raise ValueError("hardtanh partition has more than one output node")
            hardtanh_node = hardtanh_partition.output_nodes[0]

            if not isinstance(hardtanh_node, Node):
                raise ValueError(f"{hardtanh_node} is not a Node")
            if hardtanh_node.op != "call_function" or hardtanh_node.target not in [
                torch.ops.aten.hardtanh.default,
                torch.ops.aten.hardtanh_.default,
            ]:
                raise ValueError(f"{hardtanh_node} is not an aten hardtanh operator")

            if _is_annotated([hardtanh_node]):
                continue

            input_qspec_map = {}
            input_act = hardtanh_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = qconfig.input_activation

            hardtanh_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_mean(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.mean]
        qconfig = self._get_quant_config(op_sources)
        mean_partitions = get_source_partitions(gm.graph, op_sources)

        mean_partitions = list(itertools.chain(*mean_partitions.values()))
        for mean_partition in mean_partitions:
            if len(mean_partition.output_nodes) > 1:
                raise ValueError("mean partition has more than one output node")
            mean_node = mean_partition.output_nodes[0]
            if _is_annotated([mean_node]):
                continue

            mean_input_node = mean_node.args[0]
            assert isinstance(mean_input_node, Node)
            mean_input_qspec_map = {}
            mean_input_qspec_map[mean_input_node] = qconfig.input_activation

            mean_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=mean_input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_unsqueeze(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [
            torch.ops.aten.unsqueeze_copy.default,
            torch.unsqueeze,
            "unsqueeze",
        ]
        qconfig = self._get_quant_config(op_sources)
        unsqueeze_partitions = get_source_partitions(gm.graph, op_sources)

        unsqueeze_partitions = list(itertools.chain(*unsqueeze_partitions.values()))
        for unsqueeze_partition in unsqueeze_partitions:
            if len(unsqueeze_partition.output_nodes) > 1:
                raise ValueError("unsqueeze partition has more than one output node")
            unsqueeze_node = unsqueeze_partition.output_nodes[0]
            if _is_annotated([unsqueeze_node]):
                continue

            unsqueeze_input_node = unsqueeze_node.args[0]
            assert isinstance(unsqueeze_input_node, Node)
            unsqueeze_input_qspec_map = {}
            unsqueeze_input_qspec_map[unsqueeze_input_node] = qconfig.input_activation

            unsqueeze_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=unsqueeze_input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_flatten(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.ops.aten.flatten.using_ints, torch.flatten]
        qconfig = self._get_quant_config(op_sources)
        if qconfig and qconfig != self.global_quant_config:
            raise NotImplementedError(
                "Havn't done custom annotation for input_out_obs_sharing_op yet"
            )
        self._annotate_input_out_obs_sharing_op(op_sources, gm)

    def _annotate_maxpool2d(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.nn.MaxPool2d,
        ]
        qconfig = self._get_quant_config(op_sources)
        maxpool_partitions = get_source_partitions(gm.graph, op_sources)

        maxpool_partitions = list(itertools.chain(*maxpool_partitions.values()))
        for maxpool_partition in maxpool_partitions:
            if len(maxpool_partition.output_nodes) > 1:
                raise ValueError("maxpool partition has more than one output node")
            maxpool_node = maxpool_partition.output_nodes[0]
            if _is_annotated([maxpool_node]):
                continue

            maxpool_input_node = maxpool_node.args[0]
            assert isinstance(maxpool_input_node, Node)
            maxpool_input_qspec_map = {}
            maxpool_input_qspec_map[maxpool_input_node] = qconfig.input_activation

            maxpool_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=maxpool_input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_adaptive_avgpool2d(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.nn.functional.adaptive_avg_pool2d,
            torch.nn.AdaptiveAvgPool2d,
        ]
        qconfig = self._get_quant_config(op_sources)
        adaptive_avgpool_partitions = get_source_partitions(gm.graph, op_sources)

        adaptive_avgpool_partitions = list(
            itertools.chain(*adaptive_avgpool_partitions.values())
        )
        for adaptive_avgpool_partition in adaptive_avgpool_partitions:
            adaptive_avgpool_node = adaptive_avgpool_partition.output_nodes[0]
            if _is_annotated([adaptive_avgpool_node]):
                continue

            input_node = adaptive_avgpool_node.args[0]
            assert isinstance(input_node, Node)
            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            adaptive_avgpool_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_avgpool2d(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.ops.aten.avg_pool2d.default, torch.nn.AvgPool2d]
        qconfig = self._get_quant_config(op_sources)
        avgpool_partitions = get_source_partitions(gm.graph, op_sources)

        avgpool_partitions = list(itertools.chain(*avgpool_partitions.values()))
        for avgpool_partition in avgpool_partitions:
            avgpool_node = avgpool_partition.output_nodes[0]
            if _is_annotated([avgpool_node]):
                continue

            input_node = avgpool_node.args[0]
            assert isinstance(input_node, Node)
            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            avgpool_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_cat(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.ops.aten.cat.default, torch.cat]
        qconfig = self._get_quant_config(op_sources)
        cat_partitions = get_source_partitions(gm.graph, op_sources)

        cat_partitions = list(itertools.chain(*cat_partitions.values()))
        for cat_partition in cat_partitions:
            cat_node = cat_partition.output_nodes[0]
            input_nodes = cat_node.args[0]
            if _is_annotated([cat_node]):
                continue

            assert isinstance(input_nodes, Sequence)

            first_input_node = input_nodes[0]
            input_qspec_map = {}
            assert isinstance(first_input_node, Node)
            assert isinstance(cat_node, Node)
            input_qspec_map[first_input_node] = qconfig.input_activation
            share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
                (first_input_node, cat_node)
            )

            for input_node in input_nodes[1:]:
                if input_node not in input_qspec_map:
                    assert isinstance(input_node, Node)
                    input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

            cat_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=share_qparams_with_input_act0_qspec,
                _annotated=True,
            )

    def _annotate_upsample2d(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.functional.interpolate, torch.nn.UpsamplingBilinear2d]
        qconfig = self._get_quant_config(op_sources)
        upsample_partitions = get_source_partitions(gm.graph, op_sources)

        upsample_partitions = list(itertools.chain(*upsample_partitions.values()))
        for upsample_partition in upsample_partitions:
            upsample_node = upsample_partition.output_nodes[0]
            input_node = upsample_node.args[0]
            if _is_annotated([upsample_node]):
                continue

            input_node = upsample_node.args[0]
            assert isinstance(input_node, Node)
            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            upsample_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_embedding(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.modules.sparse.Embedding, torch.nn.modules.Embedding]
        qconfig = self._get_quant_config(op_sources)
        embedding_partitions = get_source_partitions(gm.graph, op_sources)

        embedding_partitions = list(itertools.chain(*embedding_partitions.values()))
        for embedding_partition in embedding_partitions:
            embedding_node = embedding_partition.output_nodes[0]

            weight = embedding_node.args[0]

            input_qspec_map = {}
            input_qspec_map[weight] = qconfig.input_activation

            embedding_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((weight, embedding_node)),
                _annotated=True,
            )

    def _annotate_softmax(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.nn.Softmax, F.softmax]
        qconfig = self._get_quant_config(op_sources)
        softmax_partitions = get_source_partitions(gm.graph, op_sources)

        softmax_partitions = list(itertools.chain(*softmax_partitions.values()))
        for softmax_partition in softmax_partitions:
            softmax_node = softmax_partition.output_nodes[0]
            input_node = softmax_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            softmax_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=qconfig.output_activation,
                _annotated=True,
            )

    def _annotate_pad(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [F.pad]
        qconfig = self._get_quant_config(op_sources)
        pad_partitions = get_source_partitions(gm.graph, op_sources)

        pad_partitions = list(itertools.chain(*pad_partitions.values()))
        for pad_partition in pad_partitions:
            pad_node = pad_partition.output_nodes[0]
            input_node = pad_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            pad_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, pad_node)),
                _annotated=True,
            )

    def _annotate_reshape(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [torch.reshape, "reshape"]
        qconfig = self._get_quant_config(op_sources)
        reshape_partitions = get_source_partitions(gm.graph, op_sources)

        reshape_partitions = list(itertools.chain(*reshape_partitions.values()))
        for reshape_partition in reshape_partitions:
            reshape_node = reshape_partition.output_nodes[0]
            input_node = reshape_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            reshape_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, reshape_node)),
                _annotated=True,
            )

    def _annotate_stride(self, gm: torch.fx.GraphModule) -> None:
        op_sources = ["stride"]
        qconfig = self._get_quant_config(op_sources)
        stride_partitions = get_source_partitions(gm.graph, op_sources)

        stride_partitions = list(itertools.chain(*stride_partitions.values()))
        for stride_partition in stride_partitions:
            stride_node = stride_partition.output_nodes[0]
            input_node = stride_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            stride_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, stride_node)),
                _annotated=True,
            )

    def _annotate_get_item(self, gm: torch.fx.GraphModule) -> None:
        op_sources = [operator.getitem]
        qconfig = self._get_quant_config(op_sources)

        get_item_partitions = get_source_partitions(gm.graph, op_sources)

        get_item_partitions = list(itertools.chain(*get_item_partitions.values()))
        for get_item_partition in get_item_partitions:
            for node in get_item_partition.nodes:
                quantization_annotation = node.meta.get(QUANT_ANNOTATION_KEY, None)
                if len(node.args) > 0 and not quantization_annotation:
                    input_node = node.args[0]
                    if (
                        input_node.op == "get_attr"
                        and getattr(gm, input_node.target).dtype == torch.int64
                    ):
                        break
                    if input_node.meta["val"].dtype == torch.int64:
                        break

                    input_node_quantization_annotation = input_node.meta.get(
                        QUANT_ANNOTATION_KEY, None
                    )
                    input_qspec_map = {}

                    if not input_node_quantization_annotation:
                        input_qspec_map[input_node] = qconfig.input_activation
                    else:
                        input_qspec_map[input_node] = SharedQuantizationSpec(input_node)

                    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=SharedQuantizationSpec((input_node, node)),
                        _annotated=True,
                    )

    def _annotate_view(self, gm: torch.fx.GraphModule) -> None:
        op_sources = ["view"]
        qconfig = self._get_quant_config(op_sources)
        view_partitions = get_source_partitions(gm.graph, op_sources)

        view_partitions = list(itertools.chain(*view_partitions.values()))
        for view_partition in view_partitions:
            view_node = view_partition.output_nodes[0]
            input_node = view_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            view_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, view_node)),
                _annotated=True,
            )

    def _annotate_permute(self, gm: torch.fx.GraphModule) -> None:
        op_sources = ["permute"]
        qconfig = self._get_quant_config(op_sources)
        permute_partitions = get_source_partitions(gm.graph, op_sources)

        permute_partitions = list(itertools.chain(*permute_partitions.values()))
        for permute_partition in permute_partitions:
            permute_node = permute_partition.output_nodes[0]
            input_node = permute_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            permute_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, permute_node)),
                _annotated=True,
            )

    def _annotate_transpose(self, gm: torch.fx.GraphModule) -> None:
        op_sources = ["transpose"]
        qconfig = self._get_quant_config(op_sources)
        transpose_partitions = get_source_partitions(gm.graph, op_sources)

        transpose_partitions = list(itertools.chain(*transpose_partitions.values()))
        for transpose_partition in transpose_partitions:
            transpose_node = transpose_partition.output_nodes[0]
            input_node = transpose_node.args[0]

            input_qspec_map = {}
            input_qspec_map[input_node] = qconfig.input_activation

            transpose_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input_node, transpose_node)),
                _annotated=True,
            )

    def _preprocess(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        model = RemoveClone()(model).graph_module
        model = ConvertHardsigmoid(quantization_capture=True)(model).graph_module
        model = ReduceDynamicRange()(model).graph_module
        return model

    def _annotate_custom_annotation(self, gm: torch.fx.GraphModule) -> None:
        for annotation_func in self.custom_quant_annotations:
            annotation_func(gm)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        model = self._preprocess(model)
        self._annotate_conv2d(model)
        self._annotate_relu(model)
        self._annotate_linear(model)
        self._annotate_binary(model)
        self._annotate_hardswish(model)
        self._annotate_hardsigmoid(model)
        self._annotate_hardtanh(model)
        self._annotate_mean(model)
        self._annotate_maxpool2d(model)
        self._annotate_avgpool2d(model)
        self._annotate_adaptive_avgpool2d(model)
        self._annotate_upsample2d(model)
        self._annotate_unsqueeze(model)
        self._annotate_flatten(model)
        self._annotate_embedding(model)
        self._annotate_cat(model)
        self._annotate_softmax(model)
        self._annotate_pad(model)
        self._annotate_reshape(model)
        self._annotate_stride(model)
        self._annotate_get_item(model)
        self._annotate_view(model)
        self._annotate_permute(model)
        self._annotate_transpose(model)
        self._annotate_matmul(model)
        self._annotate_custom_annotation(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
