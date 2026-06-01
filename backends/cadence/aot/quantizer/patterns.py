# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.cadence.aot.compiler_utils import get_shape
from executorch.backends.cadence.aot.pass_utils import get_arg, replace_with_op
from executorch.backends.cadence.aot.quantizer.pattern_utils import (
    DQ_PER_TENSOR,
    find_quant_user,
    fuse_conv,
    fuse_linear,
    fuse_matmul,
    insert_node_with_meta,
)
from executorch.backends.cadence.aot.quantizer.utils import (
    check_out_zero_point_is_min_range,
    get_bias_qparams,
    quantize_tensor_multiplier,
)
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

    def anchor_ops(self) -> tuple[OpOverload, ...]:
        return tuple(self.partition_types())

    def fuse(
        self,
        gm: fx.GraphModule,
        anchor_node: fx.Node,
    ) -> Optional[fx.Node]:
        """Replace the dq→op→q subgraph around ``anchor_node`` with a fused op.

        Called by ``QuantFusionPass`` for each node matching ``anchor_ops()``.
        Returns the new fused node on success, or ``None`` to skip this match.
        Subclasses override to implement pattern-specific fusion logic.
        """
        return None


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
        return torch.ops.cadence.quantized_linear.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        assert anchor_node.target == torch.ops.aten.addmm.default
        # addmm(bias, input, weight)
        bias_node = anchor_node.args[0]
        assert isinstance(bias_node, fx.Node)
        dq_input = get_arg(anchor_node, "mat1", fx.Node)
        if dq_input.target != DQ_PER_TENSOR:
            return None
        dq_weight = get_arg(anchor_node, "mat2", fx.Node)
        if dq_weight.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        dq_bias = bias_node if bias_node.target == DQ_PER_TENSOR else None
        weight_q = get_arg(dq_weight, "input", fx.Node)
        transposed = insert_node_with_meta(
            gm,
            torch.ops.aten.transpose.int,
            (weight_q, 0, 1),
            None,
            anchor_node,
            weight_q,
        )
        return fuse_linear(
            gm,
            dq_input,
            dq_weight,
            dq_bias,
            quant_node,
            anchor_node,
            self.replacement_op(),
            weight_q=transposed,
        )


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
        return torch.ops.cadence.quantized_add.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        # Skip if alpha kwarg is present — changes add semantics.
        if anchor_node.kwargs:
            return None
        dq0 = anchor_node.args[0]
        if not isinstance(dq0, fx.Node) or dq0.target != DQ_PER_TENSOR:
            return None
        dq1 = anchor_node.args[1]
        if not isinstance(dq1, fx.Node) or dq1.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        args = (
            get_arg(dq0, "input", fx.Node),
            get_arg(dq0, "scale", float),
            get_arg(dq0, "zero_point", int),
            get_arg(dq1, "input", fx.Node),
            get_arg(dq1, "scale", float),
            get_arg(dq1, "zero_point", int),
            get_arg(quant_node, "scale", float),
            get_arg(quant_node, "zero_point", int),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, quant_node
        )


# This is a base class for Add+ReLU fusion, since it can be used with two different relu aten ops
class AddReluBasePattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> List[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # The first node should be add, the second should be relu
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        add_node = fused_partition[0].nodes[-1]
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        relu_node = fused_partition[1].nodes[-1]

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
                output=[(relu_node,)],  # Output is from the relu node
            ),
            relu_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_add.per_tensor

    def anchor_ops(self) -> tuple[OpOverload, ...]:
        return (torch.ops.aten.add.Tensor,)

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        add_users = list(anchor_node.users)
        if len(add_users) != 1:
            return None
        relu_node = add_users[0]
        if relu_node.target != self.partition_types()[1]:
            return None
        if len(anchor_node.kwargs) > 0:
            return None
        dq0 = anchor_node.args[0]
        if not isinstance(dq0, fx.Node) or dq0.target != DQ_PER_TENSOR:
            return None
        dq1 = anchor_node.args[1]
        if not isinstance(dq1, fx.Node) or dq1.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(relu_node)
        if quant_node is None:
            return None
        if not check_out_zero_point_is_min_range(
            get_arg(quant_node, "zero_point", int),
            get_arg(quant_node, "dtype", torch.dtype),
        ):
            return None
        args = (
            get_arg(dq0, "input", fx.Node),
            get_arg(dq0, "scale", float),
            get_arg(dq0, "zero_point", int),
            get_arg(dq1, "input", fx.Node),
            get_arg(dq1, "scale", float),
            get_arg(dq1, "zero_point", int),
            get_arg(quant_node, "scale", float),
            get_arg(quant_node, "zero_point", int),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, quant_node
        )


# Add + regular relu op fusion
class AddReluPattern0(AddReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.add.Tensor, torch.ops.aten.relu.default]


# Add + alternate relu op fusion
class AddReluPattern1(AddReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.add.Tensor, torch.ops.aten.relu_.default]


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
        # TODO: T240804887 This is actually a per-tensor variant,
        # we just need to change the name of the op
        return torch.ops.cadence.quantized_matmul.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq0 = anchor_node.args[0]
        if not isinstance(dq0, fx.Node) or dq0.target != DQ_PER_TENSOR:
            return None
        dq1 = anchor_node.args[1]
        if not isinstance(dq1, fx.Node) or dq1.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        return fuse_matmul(gm, anchor_node, dq0, dq1, quant_node, self.replacement_op())


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

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        cat_inputs = anchor_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or not cat_inputs:
            return None
        inputs_q = []
        for inp in cat_inputs:
            if not isinstance(inp, fx.Node) or inp.target != DQ_PER_TENSOR:
                return None
            inputs_q.append(get_arg(inp, "input", fx.Node))
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        dim = get_arg(anchor_node, "dim", int)
        args = (inputs_q,)
        kwargs = {"dim": dim}
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, kwargs, quant_node
        )


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
        return torch.ops.cadence.quantized_conv1d_ncl.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        dq_weight = anchor_node.args[1]
        if not isinstance(dq_weight, fx.Node) or dq_weight.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        return fuse_conv(self, gm, anchor_node, dq_input, dq_weight, quant_node)


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
        return torch.ops.cadence.quantized_conv2d_nchw.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        dq_weight = anchor_node.args[1]
        if not isinstance(dq_weight, fx.Node) or dq_weight.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        return fuse_conv(self, gm, anchor_node, dq_input, dq_weight, quant_node)


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
        return torch.ops.cadence.quantized_layer_norm.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        scale = get_arg(dq_input, "scale", float)
        zero_point = get_arg(dq_input, "zero_point", int)
        normalized_shape = anchor_node.args[1]
        assert isinstance(normalized_shape, list)
        weight = (
            anchor_node.args[2]
            if len(anchor_node.args) > 2 and anchor_node.args[2]
            else None
        )
        bias = (
            anchor_node.args[3]
            if len(anchor_node.args) > 3 and anchor_node.args[3]
            else None
        )
        input_q = get_arg(dq_input, "input", fx.Node)
        # Default weight=1 and bias=0 must be float32 — cadence::quantized_layer_norm
        # expects float affine parameters, not quantized values.
        if not weight:
            weight = insert_node_with_meta(
                gm,
                torch.ops.aten.full.default,
                (normalized_shape, 1),
                {"dtype": torch.float32},
                anchor_node,
                input_q,
            )
        if not bias:
            bias = insert_node_with_meta(
                gm,
                torch.ops.aten.full.default,
                (normalized_shape, 0),
                {"dtype": torch.float32},
                anchor_node,
                input_q,
            )
        args = (input_q, scale, zero_point)
        kwargs = {
            "normalized_shape": normalized_shape,
            "weight": weight,
            "bias": bias,
            "eps": get_arg(anchor_node, "eps", float),
            "output_scale": get_arg(quant_node, "scale", float),
            "output_zero_point": get_arg(quant_node, "zero_point", int),
        }
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, kwargs, quant_node
        )


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
        return torch.ops.cadence.quantized_linear.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        dq_weight = anchor_node.args[1]
        if not isinstance(dq_weight, fx.Node) or dq_weight.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        dq_bias: fx.Node | None = None
        if len(anchor_node.args) > 2:
            bias_arg = anchor_node.args[2]
            if isinstance(bias_arg, fx.Node) and bias_arg.target == DQ_PER_TENSOR:
                dq_bias = bias_arg
        return fuse_linear(
            gm,
            dq_input,
            dq_weight,
            dq_bias,
            quant_node,
            anchor_node,
            self.replacement_op(),
        )


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
        # TODO: T240804887 This is actually a per-tensor variant, we just need to change the name of the op
        return torch.ops.cadence.quantized_matmul.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq0 = anchor_node.args[0]
        if not isinstance(dq0, fx.Node) or dq0.target != DQ_PER_TENSOR:
            return None
        dq1 = anchor_node.args[1]
        if not isinstance(dq1, fx.Node) or dq1.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        return fuse_matmul(gm, anchor_node, dq0, dq1, quant_node, self.replacement_op())


class MaxPool2dPattern(QuantizationPattern):
    """
    Pattern for quantized max pooling (with indices variant).

    Max pooling is order-preserving, so max(a, b) in the quantized domain gives
    the same result as quantizing max(dequant(a), dequant(b)) when using the same
    scale/zero_point. This means we can perform max pooling directly on quantized
    values without any requantization.

    The input and output share quantization parameters.
    """

    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.max_pool2d_with_indices.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        max_pool_node = fused_partition[0].nodes[-1]

        # Since max_pool2d_with_indices returns a tuple, the output observer must be
        # placed on getitem[0] rather than the tuple-returning op. Otherwise
        # prepare_pt2e silently skips it.
        # Expect exactly one user: getitem[0] extracting the values tensor. If indices
        # are also used or the structure is unexpected, bail out.
        users = list(max_pool_node.users)
        if (
            len(users) != 1
            or users[0].target is not operator.getitem
            or users[0].args[1] != 0
        ):
            return PartitionAnchors(empty=True), max_pool_node
        getitem_0 = users[0]

        return (
            PartitionAnchors(
                inputs=[(max_pool_node, 0)],
                weights=[],
                biases=[],
                # kernel_size, stride, padding, dilation, ceil_mode are literals
                literals=[
                    (max_pool_node, i) for i in range(1, len(max_pool_node.args))
                ],
                output=[
                    (
                        getitem_0,
                        SharedQuantizationSpec((max_pool_node.args[0], max_pool_node)),
                    )
                ],
            ),
            max_pool_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_max_pool2d_nchw.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        return _fuse_max_pool2d(gm, anchor_node)


def _fuse_max_pool2d(gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
    """Shared fuse logic for both MaxPool2d variants."""
    dq_input = anchor_node.args[0]
    if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
        return None
    quant_node = find_quant_user(anchor_node)
    if quant_node is None:
        return None
    kernel_size = get_arg(anchor_node, "kernel_size", list[int])
    stride = get_arg(anchor_node, "stride", list[int])
    padding = get_arg(anchor_node, "padding", list[int])
    dilation = get_arg(anchor_node, "dilation", list[int])
    ceil_mode = get_arg(anchor_node, "ceil_mode", bool)
    args = (get_arg(dq_input, "input", fx.Node),)
    kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }
    return replace_with_op(
        gm,
        anchor_node,
        torch.ops.cadence.quantized_max_pool2d_nchw.default,
        args,
        kwargs,
        quant_node,
    )


class MaxPool2dWithoutIndicesPattern(QuantizationPattern):
    """
    Pattern for quantized max pooling (without indices variant).

    Same as MaxPool2dPattern but matches aten.max_pool2d.default which returns
    a single tensor instead of a tuple (values, indices).
    """

    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.max_pool2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        max_pool_node = fused_partition[0].nodes[-1]

        return (
            PartitionAnchors(
                inputs=[(max_pool_node, 0)],
                weights=[],
                biases=[],
                literals=[
                    (max_pool_node, i) for i in range(1, len(max_pool_node.args))
                ],
                output=[
                    (
                        max_pool_node,
                        SharedQuantizationSpec((max_pool_node.args[0], max_pool_node)),
                    )
                ],
            ),
            max_pool_node,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_max_pool2d_nchw.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        return _fuse_max_pool2d(gm, anchor_node)


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
        return torch.ops.cadence.quantized_relu.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        input_scale = get_arg(dq_input, "scale", float)
        requantize_scale = input_scale / get_arg(quant_node, "scale", float)
        requantize_scale_t = torch.tensor([requantize_scale])
        out_multiplier, out_shift = quantize_tensor_multiplier(requantize_scale_t)
        args = (get_arg(dq_input, "input", fx.Node),)
        kwargs = {
            "X_zero_point": get_arg(dq_input, "zero_point", int),
            "out_zero_point": get_arg(quant_node, "zero_point", int),
            "out_multiplier": out_multiplier[0].item(),
            "out_shift": out_shift[0].item(),
        }
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, kwargs, quant_node
        )


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
        return torch.ops.cadence.quantized_conv2d_nchw.per_tensor

    def anchor_ops(self) -> tuple[OpOverload, ...]:
        return (self.partition_types()[0],)

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        conv_users = list(anchor_node.users)
        if len(conv_users) != 1:
            return None
        relu_node = conv_users[0]
        if relu_node.target != self.partition_types()[1]:
            return None
        _arg0 = anchor_node.args[0]
        dq_input = (
            _arg0
            if isinstance(_arg0, fx.Node) and _arg0.target == DQ_PER_TENSOR
            else None
        )
        _arg1 = anchor_node.args[1]
        dq_weight = (
            _arg1
            if isinstance(_arg1, fx.Node) and _arg1.target == DQ_PER_TENSOR
            else None
        )
        if dq_input is None or dq_weight is None:
            return None
        quant_node = find_quant_user(relu_node)
        if quant_node is None:
            return None
        check_out_zero_point_is_min_range(
            get_arg(quant_node, "zero_point", int),
            get_arg(quant_node, "dtype", torch.dtype),
        )
        return fuse_conv(self, gm, anchor_node, dq_input, dq_weight, quant_node)


# Conv1d + regular relu op fusion
class Conv1dReluPattern0(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default, torch.ops.aten.relu.default]

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv1d_ncl.per_tensor


# Conv1d + alternate relu op fusion
class Conv1dReluPattern1(ConvReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default, torch.ops.aten.relu_.default]

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv1d_ncl.per_tensor


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
        return torch.ops.cadence.quantized_softmax.per_tensor

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        dq_input = anchor_node.args[0]
        if not isinstance(dq_input, fx.Node) or dq_input.target != DQ_PER_TENSOR:
            return None
        quant_node = find_quant_user(anchor_node)
        if quant_node is None:
            return None
        input_q = get_arg(dq_input, "input", fx.Node)
        quant_input = get_arg(quant_node, "input", fx.Node)
        mask_shape = get_shape(gm, quant_input)
        if not mask_shape:
            return None
        mask_shape = list(mask_shape)
        # Softmax mask is packed 16 elements per int32 word.
        mask_shape[-1] = mask_shape[-1] // 16
        mask_tensor = insert_node_with_meta(
            gm,
            torch.ops.aten.full.default,
            (mask_shape, 0.0),
            {"dtype": torch.int32},
            anchor_node,
            input_q,
        )
        # Initial position for streaming softmax (unused, set to 0).
        pos_tensor = insert_node_with_meta(
            gm,
            torch.ops.aten.full.default,
            ([1], 0),
            {"dtype": torch.int64},
            anchor_node,
            input_q,
        )
        args = (
            input_q,
            mask_tensor,
            get_arg(anchor_node, "dim", int),
            0,
            pos_tensor,
            get_arg(dq_input, "scale", float),
            get_arg(dq_input, "zero_point", int),
            get_arg(quant_node, "scale", float),
            get_arg(quant_node, "zero_point", int),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, quant_node
        )


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

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        if len(anchor_node.args) != 3 or len(anchor_node.kwargs) > 0:
            return None
        _arg1 = anchor_node.args[1]
        dq_weight = (
            _arg1
            if isinstance(_arg1, fx.Node) and _arg1.target == DQ_PER_TENSOR
            else None
        )
        _arg2 = anchor_node.args[2]
        dq_bias = (
            _arg2
            if isinstance(_arg2, fx.Node) and _arg2.target == DQ_PER_TENSOR
            else None
        )
        if dq_weight is None or dq_bias is None:
            return None
        input_node = anchor_node.args[0]
        assert isinstance(input_node, fx.Node)
        args = (
            input_node,
            get_arg(dq_weight, "input", fx.Node),
            get_arg(dq_weight, "scale", float),
            get_arg(dq_bias, "input", fx.Node),
            get_arg(dq_bias, "scale", float),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, anchor_node
        )


class MixedW8A32ConvPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-ignore[29]
        conv_layer = fused_partition[0].nodes[-1]

        # Bail if the arguments have different shapes than expected
        # Stride, padding, dilation and groups are not supported
        if len(conv_layer.args) != 3 or len(conv_layer.kwargs) > 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                conv_layer,
            )

        cnn_weights = conv_layer.args[1]
        if "tensor_meta" in cnn_weights.meta:
            cnn_weights_shape = cnn_weights.meta["tensor_meta"].shape
            # Bail if the channels are not multiple of 4 (SIMD)
            if cnn_weights_shape[0] % 4 != 0:
                return (
                    PartitionAnchors(
                        empty=True,
                    ),
                    conv_layer,
                )
            if cnn_weights_shape[1] % 4 != 0:
                return (
                    PartitionAnchors(
                        empty=True,
                    ),
                    conv_layer,
                )
            # Bail if the kernel size is not 3
            if cnn_weights_shape[2] != 3:
                return (
                    PartitionAnchors(
                        empty=True,
                    ),
                    conv_layer,
                )

            inputs = conv_layer.args[0]
            if "tensor_meta" in inputs.meta:
                inputs_shape = inputs.meta["tensor_meta"].shape
                # Bail if length != kernel size - Not yet supported
                if inputs_shape[-1] != cnn_weights_shape[2]:
                    return (
                        PartitionAnchors(
                            empty=True,
                        ),
                        conv_layer,
                    )

        return (
            PartitionAnchors(
                inputs=[],
                weights=[(conv_layer, 1)],
                biases=[(conv_layer, 2)],
                output=[],
                others=[(conv_layer, 0)],
            ),
            conv_layer,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_w8a32_conv.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        if len(anchor_node.args) != 3 or len(anchor_node.kwargs) > 0:
            return None
        _arg1 = anchor_node.args[1]
        dq_weight = (
            _arg1
            if isinstance(_arg1, fx.Node) and _arg1.target == DQ_PER_TENSOR
            else None
        )
        _arg2 = anchor_node.args[2]
        dq_bias = (
            _arg2
            if isinstance(_arg2, fx.Node) and _arg2.target == DQ_PER_TENSOR
            else None
        )
        if dq_weight is None or dq_bias is None:
            return None
        input_node = anchor_node.args[0]
        assert isinstance(input_node, fx.Node)
        assert get_arg(anchor_node, "stride", list[int]) == [1]
        assert get_arg(anchor_node, "padding", list[int]) == [0]
        assert get_arg(anchor_node, "dilation", list[int]) == [1]
        assert get_arg(anchor_node, "groups", int) == 1
        weight_q = get_arg(dq_weight, "input", fx.Node)
        transposed_inputs = insert_node_with_meta(
            gm,
            torch.ops.aten.permute.default,
            (input_node, [0, 2, 1]),
            None,
            anchor_node,
            input_node,
        )
        transposed_weights = insert_node_with_meta(
            gm,
            torch.ops.aten.permute.default,
            (weight_q, [2, 0, 1]),
            None,
            anchor_node,
            weight_q,
        )
        args = (
            transposed_inputs,
            transposed_weights,
            get_arg(dq_weight, "scale", float),
            get_arg(dq_bias, "input", fx.Node),
            get_arg(dq_bias, "scale", float),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, anchor_node
        )


class MixedW8A32GruPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.gru.input]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        gru_layer = fused_partition[0].nodes[-1]
        if len(gru_layer.kwargs) > 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                gru_layer,
            )

        # Bail if input or states are not multiple of 4 (SIMD)
        tensor_meta_0 = gru_layer.args[0].meta.get("tensor_meta", None)
        if tensor_meta_0 is None or tensor_meta_0.shape[-1] % 4 != 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                gru_layer,
            )
        tensor_meta_1 = gru_layer.args[1].meta.get("tensor_meta", None)
        if tensor_meta_1 is None or tensor_meta_1.shape[-1] % 4 != 0:
            return (
                PartitionAnchors(
                    empty=True,
                ),
                gru_layer,
            )

        class Wrapper:  # noqa: B903
            def __init__(self, args, meta):
                self.args = args
                self.meta = meta

        wrapper = Wrapper(tuple(gru_layer.args[2]), gru_layer.meta)

        # Using SharedQuantizationSpec so that bias_hh has the same observer as bias_ih
        # Both biases get the same quantization scale to match the cpp operator
        bias_ih_node = wrapper.args[2]
        bias_ih_edge = (bias_ih_node, gru_layer)
        shared_bias_qspec = SharedQuantizationSpec(edge_or_node=bias_ih_edge)

        return (
            PartitionAnchors(
                inputs=[],
                # pyre-fixme[6]: Expected `List[Tuple[Node, int]]` but got `List[Tuple[Wrapper, int]]`.
                weights=[(wrapper, 0), (wrapper, 1)],
                # pyre-fixme[6]: Expected `List[Union[Tuple[Node, int], Tuple[Node, int, DerivedQuantizationSpec]]]` but got `List[Tuple[Wrapper, int]]`.
                biases=[
                    (wrapper, 2),  # bias_ih gets normal qspec
                    (
                        wrapper,
                        3,
                        shared_bias_qspec,
                    ),  # bias_hh shares observer with bias_ih
                ],
                output=[],
                others=[(gru_layer, 0), (gru_layer, 1)],
            ),
            gru_layer,
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_w8a32_gru.default

    def fuse(self, gm: fx.GraphModule, anchor_node: fx.Node) -> fx.Node | None:
        if len(anchor_node.kwargs) > 0:
            return None
        params = anchor_node.args[2]
        # GRU requires 4 weight/bias params: w_ih, w_hh, b_ih, b_hh
        if not isinstance(params, (list, tuple)) or len(params) < 4:
            return None
        dq_w_ih = params[0]
        if not isinstance(dq_w_ih, fx.Node) or dq_w_ih.target != DQ_PER_TENSOR:
            return None
        dq_w_hh = params[1]
        if not isinstance(dq_w_hh, fx.Node) or dq_w_hh.target != DQ_PER_TENSOR:
            return None
        dq_b_ih = params[2]
        if not isinstance(dq_b_ih, fx.Node) or dq_b_ih.target != DQ_PER_TENSOR:
            return None
        dq_b_hh = params[3]
        if not isinstance(dq_b_hh, fx.Node) or dq_b_hh.target != DQ_PER_TENSOR:
            return None
        input_node = anchor_node.args[0]
        hidden_node = anchor_node.args[1]
        args = (
            input_node,
            hidden_node,
            get_arg(dq_w_ih, "input", fx.Node),
            get_arg(dq_w_ih, "scale", float),
            get_arg(dq_w_hh, "input", fx.Node),
            get_arg(dq_w_hh, "scale", float),
            get_arg(dq_b_ih, "input", fx.Node),
            get_arg(dq_b_ih, "scale", float),
            get_arg(dq_b_hh, "input", fx.Node),
        )
        return replace_with_op(
            gm, anchor_node, self.replacement_op(), args, {}, anchor_node
        )


class RmsNormPattern(QuantizationPattern):
    """Pattern that preserves rms_norm from decomposition without matching anything."""

    def partition_types(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.aten.rms_norm.default]

    def get_anchors(
        self, gm: torch.fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Tuple[PartitionAnchors, fx.Node]:
        return PartitionAnchors(empty=True), None  # pyre-ignore[7]

    def replacement_op(self) -> torch._ops.OpOverload:
        return torch.ops.aten.rms_norm.default
