# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numbers
import operator
from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch._ops import OpOverload

from torch._subclasses import FakeTensor
from torch.fx import Node

from torchao.quantization.pt2e import FixedQParamsFakeQuantize, FixedQParamsObserver
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from .qconfig import (
    get_16a16w_qnn_ptq_config,
    get_16a4w_qnn_qat_config,
    get_8a8w_qnn_qat_config,
    QuantizationConfig,
)


OP_ANNOTATOR: Dict[OpOverload, Callable] = {}


def register_annotator(ops: List[OpOverload]):
    def decorator(annotator: Callable):
        for op in ops:
            OP_ANNOTATOR[op] = annotator

    return decorator


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
    return node.meta["val"].dtype == torch.float32


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if Q_ANNOTATION_KEY not in node.meta:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
        node.meta[Q_ANNOTATION_KEY]._annotated = True


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
    if _is_float_tensor(node.args[0]):
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

    if _is_float_tensor(node):
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input_act, node)),
            _annotated=True,
        )


def annotate_single_in(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_qspec_map[input_act] = quantization_config.input_activation

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
    if _is_float_tensor(node.args[0]):
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

    if _is_float_tensor(node):
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.to.dtype])
def annotate_to_dtype(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.atan.default])
def annotate_atan(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.topk.default])
def annotate_topk(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return
    # We can use single_in_single_out since we don't want to quantize indices output
    annotate_single_in_single_out(node, quantization_config)


def annotate_binary(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = (
        quantization_config.output_activation if _is_float_tensor(node) else None
    )

    input_qspec_map = {}
    input_act0 = node.args[0]
    if _is_float_tensor(input_act0):
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if _is_float_tensor(input_act1):
        input_qspec_map[input_act1] = input_act_qspec

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


@register_annotator(
    [torch.ops.aten.add, torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]
)
def annotate_add(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.amax.default])
def annotate_amax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.argmax.default])
def annotate_argmax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in(node, quantization_config)


@register_annotator([torch.ops.aten.amin.default])
def annotate_amin(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.argmin.default])
def annotate_argmin(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in(node, quantization_config)


@register_annotator([torch.ops.aten.sub, torch.ops.aten.sub.Tensor])
def annotate_sub(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.eq.Tensor])
def annotate_eq(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.ne.Tensor])
def annotate_ne(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.ge.Tensor])
def annotate_ge(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.gt.Tensor])
def annotate_gt(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.le.Tensor])
def annotate_le(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.lt.Tensor])
def annotate_lt(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.masked_fill.Tensor])
def annotate_masked_fill(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    for input_node in node.args:
        assert isinstance(input_node, Node)
        if _is_float_tensor(input_node):
            input_qspec_map[input_node] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=(
            quantization_config.output_activation if _is_float_tensor(node) else None
        ),
        _annotated=True,
    )


@register_annotator(
    [torch.ops.aten.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul_.Tensor]
)
def annotate_mul(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.max.other, torch.ops.aten.maximum.default])
def annotate_max(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.max.dim])
def annotate_max_dim(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in(node, quantization_config)


@register_annotator([torch.ops.aten.min.other, torch.ops.aten.minimum.default])
def annotate_min(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.min.dim])
def annotate_min_dim(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in(node, quantization_config)


@register_annotator(
    [torch.ops.aten.div, torch.ops.aten.div.Tensor, torch.ops.aten.divide.Tensor]
)
def annotate_div(node: Node, quantization_config: QuantizationConfig) -> None:
    def _derived_inp1_const_div_quant_spec(
        node: torch.fx.Node, output_qspec: QuantizationSpec
    ) -> DerivedQuantizationSpec:
        def _derive_div_qparams_fn(
            obs_or_fqs: List,
            const_val: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inp_0_obs_or_fq = obs_or_fqs[0]
            inp_0_scale, inp_0_zp = inp_0_obs_or_fq.calculate_qparams()
            derived_scale = inp_0_scale / const_val
            return (derived_scale, inp_0_zp)

        inp_0 = node.args[0]
        const_inp_1 = node.args[1]
        _derive_div_qparams_with_const_fn = partial(
            _derive_div_qparams_fn, const_val=const_inp_1
        )

        q_min = (
            torch.iinfo(output_qspec.dtype).min
            if output_qspec.quant_min is None
            else output_qspec.quant_min
        )
        q_max = (
            torch.iinfo(output_qspec.dtype).max
            if output_qspec.quant_max is None
            else output_qspec.quant_max
        )
        return DerivedQuantizationSpec(
            derived_from=[(inp_0, node)],
            derive_qparams_fn=_derive_div_qparams_with_const_fn,
            dtype=output_qspec.dtype,
            quant_min=q_min,
            quant_max=q_max,
            ch_axis=0,
            qscheme=output_qspec.qscheme,
        )

    if [a for a in node.args if isinstance(a, Node)]:
        annotate_binary(node, quantization_config)
    # special constant divisor case
    elif isinstance(node.args[0], Node) and isinstance(node.args[1], numbers.Number):
        if _is_annotated([node]):
            return

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = _derived_inp1_const_div_quant_spec(
            node, quantization_config.output_activation
        )
        input_qspec_map = {}
        input_act0 = node.args[0]
        if _is_float_tensor(input_act0):
            input_qspec_map[input_act0] = input_act_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    else:
        raise NotImplementedError(f"No quant annotation is implemented for {node}.")


@register_annotator([torch.ops.aten.rsub.Tensor])
def annotate_rsub(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.sum.dim_IntList])
def annotate_sum(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.abs.default])
def annotate_abs(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.torch.ops.aten.arange.default,
        torch.torch.ops.aten.arange.start,
        torch.torch.ops.aten.arange.start_step,
    ]
)
def annotate_arange(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    if _is_float_tensor(node):
        # workaround for node with kwargs could not be correctly annotated
        node.kwargs = {}
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={},
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.ceil.default])
def annotate_ceil(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
    ]
)
def annotate_clamp(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.index_select.default])
def annotate_index_select(node: Node, quantization_config: QuantizationConfig) -> None:
    # args[2] = indices, which should be int
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.flip.default])
def annotate_flip(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.floor.default])
def annotate_floor(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.relu.default, torch.ops.aten.relu_.default])
def annotate_relu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.repeat.default])
def annotate_repeat(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.round.default])
def annotate_round(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.cos.default])
def annotate_cos(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.col2im.default, torch.ops.aten.im2col.default])
def annotate_col_im(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.sin.default])
def annotate_sin(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.floor_divide.default])
def annotate_floor_divide(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.scalar_tensor.default])
def annotate_scalar_tensor(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return
    if _is_float_tensor(node):
        # workaround for node with kwargs could not be correctly annotated
        node.kwargs = {}
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={},
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.tanh.default])
def annotate_tanh(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.full_like.default, torch.ops.aten.full.default])
def annotate_full(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    if _is_float_tensor(node):
        # workaround for node with kwargs could not be correctly annotated
        node.kwargs = {}
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={},
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator(
    [torch.ops.aten.hardswish.default, torch.ops.aten.hardswish_.default]
)
def annotate_hardswish(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardsigmoid_.default]
)
def annotate_hardsigmoid(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default])
def annotate_hardtanh(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.mean.default, torch.ops.aten.mean.dim])
def annotate_mean(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.max_pool2d.default])
def annotate_max_pool2d(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.max_pool2d_with_indices.default])
def annotate_max_pool2d_with_indices(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.neg.default])
def annotate_neg(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.adaptive_avg_pool1d.default,
        torch.ops.aten.adaptive_avg_pool2d.default,
    ]
)
def annotate_adaptive_avgpool(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.avg_pool2d.default])
def annotate_avgpool2d(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.permute.default])
def annotate_permute(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.prelu.default])
def annotate_prelu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    ]
)
def annotate_view(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.pixel_shuffle.default])
def annotate_pixel_shuffle_default(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.pixel_unshuffle.default])
def annotate_pixel_unshuffle_default(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.upsample_bilinear2d.vec])
def annotate_upsample_bilinear2d(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.upsample_bicubic2d.vec])
def annotate_upsample_upsample_bicubic2d(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.upsample_nearest2d.vec])
def annotate_upsample_nearest2d(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
        torch.ops.aten._safe_softmax.default,
    ]
)
def annotate_softmax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.asin.default])
def annotate_asin(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.linalg_vector_norm.default])
def annotate_linalg_vector_norm(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.log_softmax.int])
def annotate_log_softmax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.log.default])
def annotate_log(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.pad.default])
def annotate_pad(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.reshape.default, torch.ops.aten.unflatten.int])
def annotate_reshape(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.select.int])
def annotate_select(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.sign.default])
def annotate_sign(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.slice.Tensor])
def annotate_slice(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.slice_scatter.default])
def annotate_slice_scatter(node: Node, quantization_config: QuantizationConfig) -> None:
    input = node.args[0]
    value = node.args[1]

    input_qspec_map = {}
    input_qspec_map[input] = quantization_config.input_activation
    input_qspec_map[value] = SharedQuantizationSpec((input, node))

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((input, node)),
        _annotated=True,
    )


@register_annotator([torch.ops.aten.sqrt.default])
def annotate_sqrt(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.square.default])
def annotate_square(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.gelu.default])
def annotate_gelu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.scaled_dot_product_attention.default])
def annotate_scaled_dot_product_attention(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dims,
    ]
)
def annotate_squeeze(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.rms_norm.default])
def annotate_rms_norm(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[2]

    if _is_annotated([node]):
        return

    # TODO current only support 16a16w
    annotate_input_qspec_map(
        node,
        act_node,
        quantization_config.input_activation,
    )

    annotate_input_qspec_map(
        node,
        weight_node,
        quantization_config.input_activation,
    )
    nodes_to_mark_annotated = [node]
    annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)


@register_annotator([torch.ops.aten.rsqrt.default])
def annotate_rsqrt(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.sigmoid, torch.ops.aten.sigmoid.default])
def annotate_sigmoid(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    input_qspec_map[input_act] = quantization_config.input_activation

    assert isinstance(input_act, Node)
    out_qconf = quantization_config.output_activation

    q_max = (
        torch.iinfo(out_qconf.dtype).max
        if out_qconf.quant_max is None
        else out_qconf.quant_max
    )
    q_min = (
        torch.iinfo(out_qconf.dtype).min
        if out_qconf.quant_min is None
        else out_qconf.quant_min
    )

    scale = 1 / (q_max - q_min + 1)

    bias_obs_ctr = observer = FixedQParamsObserver.with_args(
        scale=scale,
        zero_point=0,
        dtype=quantization_config.output_activation.dtype,
        qscheme=torch.torch.per_tensor_affine,
        quant_max=q_max,
        quant_min=q_min,
    )
    if quantization_config in (
        get_8a8w_qnn_qat_config(),
        get_16a4w_qnn_qat_config(),
    ):
        bias_obs_ctr = FixedQParamsFakeQuantize.with_args(
            observer=observer,
            scale=scale,
            zero_point=0,
            dtype=quantization_config.output_activation.dtype,
            qscheme=torch.torch.per_tensor_affine,
            quant_max=q_max,
            quant_min=q_min,
        )

    # make sigmoid map to the range between 0~1
    out_act_quantization_spec = QuantizationSpec(
        dtype=quantization_config.output_activation.dtype,
        quant_max=q_max,
        quant_min=q_min,
        observer_or_fake_quant_ctr=bias_obs_ctr,
        qscheme=torch.torch.per_tensor_affine,
    )

    if _is_float_tensor(node):
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=out_act_quantization_spec,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.__and__.Tensor, torch.ops.aten.logical_and.default])
def annotate_and(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.bitwise_or.Tensor, torch.ops.aten.__or__.Tensor])
def annotate_bitwise_or(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.bitwise_xor.Tensor, torch.ops.aten.__xor__.Tensor])
def annotate_xor(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.pow.Tensor_Tensor])
def annotate_pow(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.unsqueeze.default])
def annotate_unsqueeze(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.unsqueeze_copy.default,
    ]
)
def annotate_unsqueeze_copy(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.transpose.int, torch.ops.aten.swapaxes.default])
def annotate_transpose(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.elu.default])
def annotate_elu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.embedding.default, torch.ops.aten.gather.default])
def annotate_embedding(node: Node, quantization_config: QuantizationConfig) -> None:
    weight = node.args[0]

    input_qspec_map = {}
    input_qspec_map[weight] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((weight, node)),
        _annotated=True,
    )


@register_annotator([torch.ops.aten.index.Tensor])
def annotate_index(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default]
)
def annotate_index_put(node: Node, quantization_config: QuantizationConfig) -> None:
    # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
    value = node.args[2]

    input_qspec_map = {}
    input_qspec_map[value] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((value, node)),
        _annotated=True,
    )


@register_annotator(
    [torch.ops.aten.index_copy.default, torch.ops.aten.index_copy_.default]
)
def annotate_index_copy(node: Node, quantization_config: QuantizationConfig) -> None:
    # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
    value = node.args[3]

    input_qspec_map = {}
    input_qspec_map[value] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((value, node)),
        _annotated=True,
    )


@register_annotator([torch.ops.aten.exp.default])
def annotate_exp(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.expand.default, torch.ops.aten.expand_as.default])
def annotate_expand(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.group_norm.default])
def annotate_group_norm(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[2]
    bias_node = None
    if len(node.args) > 2:
        bias_node = node.args[3]

    if _is_annotated([node]):
        return

    annotate_input_qspec_map(
        node,
        act_node,
        quantization_config.input_activation,
    )
    annotate_input_qspec_map(
        node,
        weight_node,
        quantization_config.weight,
    )
    nodes_to_mark_annotated = [node, weight_node]
    if bias_node:
        annotate_input_qspec_map(
            node,
            bias_node,
            quantization_config.bias,
        )
        nodes_to_mark_annotated.append(bias_node)
    annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)


@register_annotator([torch.ops.aten.flatten.using_ints])
def annotate_flatten(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.stack.default])
def annotate_stack(node: Node, quantization_config: QuantizationConfig) -> None:
    input_nodes = node.args[0]
    if _is_annotated([node]) or not _is_float_tensor(node):
        return

    assert isinstance(input_nodes, Sequence)

    first_input_node = input_nodes[0]
    input_qspec_map = {}
    assert isinstance(first_input_node, Node)
    input_qspec_map[first_input_node] = quantization_config.input_activation
    share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
        (first_input_node, node)
    )

    for input_node in input_nodes[1:]:
        if input_node not in input_qspec_map:
            assert isinstance(input_node, Node)
            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=share_qparams_with_input_act0_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.matmul.default])
def annotate_matmul(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

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
            # we should use int16 for mm / bmm instead of int4
            input_qspec_map[input_act1] = get_16a16w_qnn_ptq_config().weight
        else:
            input_qspec_map[input_act1] = input_act_qspec

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.bmm.default])
def annotate_bmm(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation

    input_qspec_map = {}
    input_act0 = node.args[0]
    if isinstance(input_act0, Node):
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if isinstance(input_act1, Node):
        # In bmm, QNN_DATATYPE_SFIXED_POINT_16 Input1 must have QNN_DATATYPE_UFIXED_POINT_16 Input0 and must be symmetric quantized.
        if input_act_qspec.dtype == torch.int32:
            # we should use int16 for mm / bmm instead of int4
            input_qspec_map[input_act1] = get_16a16w_qnn_ptq_config().weight
        else:
            input_qspec_map[input_act1] = input_act_qspec

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )

    # We use get_source_partition in pass, but it is the same source for MultiheadAttention, so we need to change its source_fn_stack.
    node.meta["source_fn_stack"] = [(node, torch.bmm)]


@register_annotator([torch.ops.aten.cdist.default])
def annotate_cdist(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.convolution.default,
    ]
)
def annotate_conv(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    # block quantization
    if quantization_config.block_size is not None:
        quantization_config.weight.observer_or_fake_quant_ctr.p.keywords.update(
            {"block_size": quantization_config.block_size}
        )

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_spec = quantization_config.input_activation
    input_qspec_map[input_act] = input_spec

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


@register_annotator([torch.ops.aten.cumsum.default])
def annotate_cumsum(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.linear.default])
def annotate_linear(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[1]
    bias_node = None

    if len(node.args) > 2:
        bias_node = node.args[2]

    if _is_annotated([node]):
        return

    # block quantization
    if quantization_config.block_size is not None:
        quantization_config.weight.observer_or_fake_quant_ctr.p.keywords.update(
            {"block_size": quantization_config.block_size}
        )

    annotate_input_qspec_map(
        node,
        act_node,
        quantization_config.input_activation,
    )
    annotate_input_qspec_map(
        node,
        weight_node,
        quantization_config.weight,
    )
    nodes_to_mark_annotated = [node, weight_node]
    if bias_node:
        if callable(quantization_config.bias):
            bias_config = quantization_config.bias(node)
        else:
            bias_config = quantization_config.bias
        annotate_input_qspec_map(node, bias_node, bias_config)
        nodes_to_mark_annotated.append(bias_node)
    annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    # We use get_source_partition in pass, but it is the same source for MultiheadAttention, so we need to change its source_fn_stack.
    node.meta["source_fn_stack"] = [(node, torch.nn.Linear)]


@register_annotator(
    [torch.ops.aten.batch_norm.default, torch.ops.aten.instance_norm.default]
)
def annotate_batch_and_instance_norm(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    act, weight, bias = node.args[0:3]
    if _is_annotated([node]):
        return

    annotated_args = [act]
    annotate_input_qspec_map(
        node,
        act,
        quantization_config.input_activation,
    )
    # QNN requires uint8 instead of int8 in 'weight' config
    if weight is not None:
        annotate_input_qspec_map(
            node,
            weight,
            quantization_config.input_activation,
        )
        annotated_args.append(weight)

    if bias is not None:
        annotate_input_qspec_map(
            node,
            bias,
            quantization_config.bias,
        )
        annotated_args.append(bias)

    annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated([node, *annotated_args])


@register_annotator([operator.getitem])
def annotate_getitem(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    if _is_float_tensor(node):
        annotate_output_qspec(node, quantization_config.output_activation)
        _mark_nodes_as_annotated([node])


@register_annotator([torch.ops.aten.layer_norm.default])
def annotate_layer_norm(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[2]
    bias_node = None
    if len(node.args) > 2:
        bias_node = node.args[3]

    if _is_annotated([node]):
        return
    input_act_qspec = quantization_config.input_activation

    annotate_input_qspec_map(
        node,
        act_node,
        input_act_qspec,
    )
    if input_act_qspec.dtype == torch.int32:
        annotate_input_qspec_map(
            node,
            weight_node,
            get_16a16w_qnn_ptq_config().weight,
        )
    else:
        annotate_input_qspec_map(
            node,
            weight_node,
            input_act_qspec,
        )
    nodes_to_mark_annotated = [node, weight_node]
    if bias_node:
        annotate_input_qspec_map(
            node,
            bias_node,
            quantization_config.bias,
        )
        nodes_to_mark_annotated.append(bias_node)
    annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)


@register_annotator([torch.ops.aten.cat.default, torch.ops.aten.concat.default])
def annotate_cat(node: Node, quantization_config: QuantizationConfig) -> None:
    input_nodes = node.args[0]
    if _is_annotated([node]) or not _is_float_tensor(node):
        return

    assert isinstance(input_nodes, Sequence)

    first_input_node = input_nodes[0]
    input_qspec_map = {}
    assert isinstance(first_input_node, Node)
    assert isinstance(node, Node)
    if _is_float_tensor(first_input_node):
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

    for input_node in input_nodes[1:]:
        if input_node not in input_qspec_map:
            assert isinstance(input_node, Node)
            if _is_float_tensor(input_node):
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=share_qparams_with_input_act0_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.unbind.int])
def annotate_unbind(node: Node, quantization_config: QuantizationConfig) -> None:
    # Seems like unbind.int can be either float or int. Only quant when input is float.
    if _is_annotated([node]) or not _is_float_tensor(node.args[0]):
        return
    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    share_qparams_with_out_node0_qspec = SharedQuantizationSpec((node.args[0], node))
    input_qspec_map[input_act] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=share_qparams_with_out_node0_qspec,
        _annotated=True,
    )

    for user in node.users:
        user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=share_qparams_with_out_node0_qspec,
            _annotated=True,
        )


@register_annotator(
    [
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.chunk.default,
    ]
)
def annotate_chunk(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_qspec_map[input_act] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        _annotated=True,
    )

    for user in node.users:
        user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.where.self])
def annotate_where(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    for input_node in node.args:
        assert isinstance(input_node, Node)
        if _is_float_tensor(input_node):
            input_qspec_map[input_node] = quantization_config.input_activation

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=(
            quantization_config.output_activation if _is_float_tensor(node) else None
        ),
        _annotated=True,
    )


@register_annotator([torch.ops.aten.zeros.default, torch.ops.aten.zeros_like.default])
def annotate_zeros(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]) or not _is_float_tensor(node):
        return

    # workaround for node with kwargs could not be correctly annotated
    node.kwargs = {}
    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map={},
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )
