# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import torch
from executorch.backends.cadence.aot.quantizer.utils import get_bias_qparams

from torch import fx
from torch._ops import OpOverload
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    SharedQuantizationSpec,
)


@dataclass
class PartitionAnchors:
    """
    All fields except output are lists of (node, args_index) pair, where node is from
    the given partition and node.args[args_index] is an input to the partition. Assumes
    a single output.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    # Inputs can share quantization parameters
    inputs: List[
        Union[
            Tuple[fx.Node, Union[int, Tuple[int, int]]],
            Tuple[
                fx.Node,
                Union[int, Tuple[int, int]],
                SharedQuantizationSpec,
            ],
        ]
    ] = field(default_factory=list)
    weights: List[Tuple[fx.Node, int]] = field(default_factory=list)
    biases: List[
        Union[Tuple[fx.Node, int], Tuple[fx.Node, int, DerivedQuantizationSpec]]
    ] = field(default_factory=list)
    others: List[Tuple[fx.Node, int]] = field(default_factory=list)
    literals: List[Tuple[fx.Node, int]] = field(default_factory=list)
    output: List[Union[Tuple[fx.Node], Tuple[fx.Node, SharedQuantizationSpec]]] = field(
        default_factory=list
    )
    empty: bool = False


class QuantizationPattern(ABC):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        """
        List of types to be passed to find_sequential_partitions_aten.
        """
        pass

    @abstractmethod
    def get_anchors(
        self, gm: torch.fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        pass

    @abstractmethod
    def replacement_op(self) -> OpOverload:
        """
        Operator (most likely a custom one) that this partition should be fused into in
        the backend. Refer to the QuantFusion pass for examples.
        """
        pass


class AddmmPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.addmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        addmm_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (addmm_node.args[1], addmm_node),
                (addmm_node.args[2], addmm_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        return (
            PartitionAnchors(
                inputs=[(addmm_node, 1)],
                weights=[(addmm_node, 2)],
                biases=[(addmm_node, 0, bias_qspec)],
                output=[(addmm_node,)],
            ),
            addmm_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_linear.default


class AddPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.add.Tensor]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        add_node = fused_partition[0].nodes[-1]

        # Bail if:
        #   - the add node is not a tensor add
        #   - the add node has kwargs (e.g. alpha)
        is_tensor_add = isinstance(add_node.args[0], fx.Node) and isinstance(
            add_node.args[1], fx.Node
        )
        if not is_tensor_add or len(add_node.kwargs) > 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                add_node,
            )

        return (
            PartitionAnchors(
                inputs=[(add_node, 0), (add_node, 1)],
                weights=[],
                biases=[],
                output=[(add_node,)],
            ),
            add_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_add.default


class BmmPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.bmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        bmm_node = fused_partition[0].nodes[-1]

        return (
            PartitionAnchors(
                inputs=[(bmm_node, 0), (bmm_node, 1)],
                weights=[],
                biases=[],
                output=[(bmm_node,)],
            ),
            bmm_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_matmul.default


class CatPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.cat.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        cat_node = fused_partition[0].nodes[-1]

        # Create args. The first argument does not have quant spec and
        # will inherit from the overall quant spec. All subsequent args
        # will share that spec.
        # Note that outpus also share that spec.
        args: List[
            Union[
                Tuple[fx.Node, Union[int, Tuple[int, int]]],
                Tuple[
                    fx.Node,
                    Union[int, Tuple[int, int]],
                    SharedQuantizationSpec,
                ],
            ]
        ] = [(cat_node, (0, 0))]
        for i in range(1, len(cat_node.args[0])):
            args.append(
                (
                    cat_node,
                    (0, i),
                    SharedQuantizationSpec((cat_node.args[0][0], cat_node)),
                )
            )

        return (
            PartitionAnchors(
                inputs=args,
                weights=[],
                biases=[],
                output=[
                    (cat_node, SharedQuantizationSpec((cat_node.args[0][0], cat_node)))
                ],
            ),
            cat_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.aten.cat.default


class Conv1dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        conv1d_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv1d_node.args[0], conv1d_node),
                (conv1d_node.args[1], conv1d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv1d_node.args) > 2 and conv1d_node.args[2] is not None:
            bias = [(conv1d_node, 2, bias_qspec)]

        return (
            PartitionAnchors(
                inputs=[(conv1d_node, 0)],
                weights=[(conv1d_node, 1)],
                # pyre-fixme[6]: Incompatible parameter type
                biases=bias,
                output=[(conv1d_node,)],
            ),
            conv1d_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv2d_nchw.default


class Conv2dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        conv2d_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv2d_node.args[0], conv2d_node),
                (conv2d_node.args[1], conv2d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv2d_node.args) > 2 and conv2d_node.args[2] is not None:
            bias = [(conv2d_node, 2, bias_qspec)]

        return (
            PartitionAnchors(
                inputs=[(conv2d_node, 0)],
                weights=[(conv2d_node, 1)],
                # pyre-fixme[6]: Incompatible parameter type
                biases=bias,
                output=[(conv2d_node,)],
            ),
            conv2d_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv2d_nchw.default


class LayerNormPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.layer_norm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        layer_norm_node = fused_partition[0].nodes[-1]

        others = [(layer_norm_node, 1)]

        # Add weights if supplied
        if len(layer_norm_node.args) > 2 and layer_norm_node.args[2]:
            others.append((layer_norm_node, 2))

        # Add bias if supplied
        if len(layer_norm_node.args) > 3 and layer_norm_node.args[3]:
            others.append((layer_norm_node, 3))

        # Weights are used in quantized mode by our kernel, so they are
        # passed in as others here along with the normalized shape.
        return (
            PartitionAnchors(
                inputs=[(layer_norm_node, 0)],
                weights=[],
                biases=[],
                # Ordering: normalized_shape, weights, bias
                others=others,
                output=[(layer_norm_node,)],
            ),
            layer_norm_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_layer_norm.default


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        linear_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (linear_node.args[0], linear_node),
                (linear_node.args[1], linear_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(linear_node.args) > 2:
            bias = [(linear_node, 2, bias_qspec)]

        return (
            PartitionAnchors(
                inputs=[(linear_node, 0)],
                weights=[(linear_node, 1)],
                # pyre-fixme[6]: Incompatible parameter type
                biases=bias,
                output=[(linear_node,)],
            ),
            linear_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_linear.default


class MatmulPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.matmul.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        matmul_node = fused_partition[0].nodes[-1]

        return (
            PartitionAnchors(
                inputs=[(matmul_node, 0), (matmul_node, 1)],
                weights=[],
                biases=[],
                output=[(matmul_node,)],
            ),
            matmul_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_matmul.default


# This is a base class for ReLU, since it can be used with two different aten ops
class ReluBasePattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> List[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        relu_node = fused_partition[0].nodes[-1]

        return (
            PartitionAnchors(
                inputs=[(relu_node, 0)],
                weights=[],
                biases=[],
                output=[(relu_node,)],
            ),
            relu_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_relu.default


# Regular relu op
class ReluPattern0(ReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.relu.default]


# Alternate relu op
class ReluPattern1(ReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.relu_.default]


# This is a base class for Conv+ReLU fusion, since it can be used with two different relu aten ops
class ConvReluBasePattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> List[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # The first node should be conv, the second should be relu
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        conv_node = fused_partition[0].nodes[-1]  # Second to last node
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        relu_node = fused_partition[1].nodes[-1]  # Last node

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, 2, bias_qspec)]

        return (
            PartitionAnchors(
                inputs=[(conv_node, 0)],
                weights=[(conv_node, 1)],
                # pyre-fixme[6]: Incompatible parameter type
                biases=bias,
                output=[(relu_node,)],  # Output is from the relu node
            ),
            relu_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv2d_nchw.default


# Conv1d + regular relu op fusion
class Conv1dReluPattern0(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default, torch.ops.aten.relu.default]


# Conv1d + alternate relu op fusion
class Conv1dReluPattern1(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default, torch.ops.aten.relu_.default]


# Conv2d + regular relu op fusion
class Conv2dReluPattern0(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv2d.default, torch.ops.aten.relu.default]


# Conv2d + alternate relu op fusion
class Conv2dReluPattern1(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv2d.default, torch.ops.aten.relu_.default]


class SoftmaxPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten._softmax.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        softmax_node = fused_partition[0].nodes[-1]

        return (
            PartitionAnchors(
                inputs=[(softmax_node, 0)],
                weights=[],
                biases=[],
                output=[(softmax_node,)],
            ),
            softmax_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_softmax.default


class MixedW8A32LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-ignore[29]
        linear_layer = fused_partition[0].nodes[-1]

        # Bail if the arguments have different shapes than expected
        if len(linear_layer.args) != 3 or len(linear_layer.kwargs) > 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                linear_layer,
            )

        input_node = linear_layer.args[0]
        input_shape = input_node.meta["tensor_meta"].shape

        # Bail if the weights are not multiple of 4 (SIMD)
        if input_shape[-1] % 4 != 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                linear_layer,
            )
        # Currenly only supporting vector-matrix multiplication
        if len(input_shape) > 0 and input_shape[-2] != 1:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                linear_layer,
            )

        return (
            PartitionAnchors(
                inputs=[],
                weights=[(linear_layer, 1)],
                biases=[(linear_layer, 2)],
                output=[],
                others=[(linear_layer, 0)],
            ),
            linear_layer,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_w8a32_linear.default
