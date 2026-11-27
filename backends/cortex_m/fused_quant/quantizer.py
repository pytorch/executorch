# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The int8 fused_quant frontend for CMSIS-NN."""

import dataclasses
import operator
from collections.abc import Iterable
from typing import cast, Optional

import executorch.backends.cortex_m.fused_quant.ops  # noqa: F401

import torch
from executorch.backends.cortex_m.passes.decompose_sdpa_pass import DecomposeSDPAPass
from executorch.backends.cortex_m.passes.matmul_to_bmm_pass import MatmulToBmmPass
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_CHANNEL_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
    SOFTMAX_OUTPUT_FIXED_QSPEC,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.frontends.fused_quant.quantizer.quantizer import (
    ConvolutionQuantizer,
    FusedQuantQuantizer,
    get_core_aten_quantizers,
    NoopQuantizer,
    OpQuantizer,
    QuantizerBase,
)
from torch import fx
from torch._ops import OpOverload
from torch.fx.node import Argument
from torchao.quantization.pt2e import ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationConfig,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_input_qspec_map,
    annotate_output_qspec,
)

_PER_CHANNEL_QSCHEMES = (torch.per_channel_symmetric, torch.per_channel_affine)


def _derive_bias_qparams(
    obs_or_fqs: list[ObserverOrFakeQuantize],
) -> tuple[torch.Tensor, torch.Tensor]:
    act_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    return (act_scale * weight_scale).to(torch.float32), torch.zeros_like(
        weight_scale, dtype=torch.int32
    )


def _annotate_bias(node: fx.Node, weight: QuantizationSpecBase | None) -> None:
    """Quantize the bias to int32 at the accumulator scale, input * weight.

    That is the scale of the kernel's multiply-adds, so it adds the bias as is.
    """
    bias: Argument = get_arg(node, "bias")
    if not isinstance(bias, fx.Node) or not isinstance(weight, QuantizationSpec):
        return
    per_channel = weight.qscheme in _PER_CHANNEL_QSCHEMES
    annotate_input_qspec_map(
        node,
        bias,
        DerivedQuantizationSpec(
            derived_from=[
                (get_arg(node, "input", fx.Node), node),
                (get_arg(node, "weight", fx.Node), node),
            ],
            derive_qparams_fn=_derive_bias_qparams,
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.int32).min,
            quant_max=torch.iinfo(torch.int32).max - 1,
            qscheme=torch.per_channel_symmetric if per_channel else None,
            # The bias is [C_out] even where the weight's channel axis is 1.
            ch_axis=0 if per_channel else None,
        ),
    )


class _QuantizedBiasOpQuantizer(OpQuantizer):
    def annotate_node(self, node: fx.Node) -> None:
        super().annotate_node(node)
        _annotate_bias(node, self.config_for(node).weight)


class _QuantizedBiasConvolutionQuantizer(ConvolutionQuantizer):
    def annotate_node(self, node: fx.Node) -> None:
        super().annotate_node(node)
        _annotate_bias(node, self.config_for(node).weight)


def _runs_on_cmsis_nn(node: fx.Node) -> bool:
    """A convolution a CMSIS-NN kernel runs: 1-d or 2-d, and if transposed,
    2-d without groups, dilation or output padding."""
    spatial_dims = len(cast(list[int], get_arg(node, "stride")))
    if spatial_dims not in (1, 2):
        return False
    return not get_arg(node, "transposed") or (
        spatial_dims == 2
        and bool(get_arg(node, "groups") == 1)
        and all(d == 1 for d in cast(list[int], get_arg(node, "dilation")))
        and all(p == 0 for p in cast(list[int], get_arg(node, "output_padding")))
    )


class _SharedQparamsQuantizer(OpQuantizer):
    """The output, and every activation after the first, share the first's qparams.

    The kernels these ops lower to take no output qparams of their own:
    ``minimum`` and ``maximum`` compare raw int8, ``pad`` copies bytes and the
    average pool writes its output on its input's grid.
    """

    def annotate_node(self, node: fx.Node) -> None:
        args: list[Argument] = [get_arg(node, name) for name in self.activation_names]
        first, *others = args
        if not isinstance(first, fx.Node):
            raise ValueError(f"Expected a tensor input for {node.name}")
        shared = SharedQuantizationSpec((first, node))
        annotate_input_qspec_map(node, first, self.config.input_activation)
        for other in others:
            # x op x is a single edge; sharing it with itself would make it its
            # own parent, which PT2E follows until the stack runs out.
            if isinstance(other, fx.Node) and other is not first:
                annotate_input_qspec_map(node, other, shared)
        annotate_output_qspec(node, shared)


_LUT_OPS = ("log", "log2", "log10", "log1p")

_aten = torch.ops.aten
_MAX_POOLS = (_aten.max_pool2d_with_indices.default, _aten.max_pool2d.default)
_SDPA = _aten.scaled_dot_product_attention.default


def _max_pool_runs_on_cmsis_nn(node: fx.Node) -> bool:
    """arm_max_pool_s8 has no dilation or ceil mode."""
    dilation: Argument = get_arg(node, "dilation")
    dilations = dilation if isinstance(dilation, (list, tuple)) else (dilation,)
    return not get_arg(node, "ceil_mode") and all(d == 1 for d in dilations)


def _pooled(node: fx.Node) -> Optional[tuple[fx.Node, fx.Node]]:
    """The max pool whose values node reads, and the pool's input."""
    source: Argument = get_arg(node, "input")
    if isinstance(source, fx.Node) and source.target is operator.getitem:
        if source.args[1] != 0:
            return None
        source = source.args[0]
    if (
        not isinstance(source, fx.Node)
        or source.target not in _MAX_POOLS
        or not _max_pool_runs_on_cmsis_nn(source)
    ):
        return None
    pool_input = source.args[0]
    return (source, pool_input) if isinstance(pool_input, fx.Node) else None


class _PooledActivationQuantizer(OpQuantizer):
    """A clamping activation reading a max pool writes the pool's grid.

    The pool writes its input's grid, so the lowering folds the activation
    into the pool only on that grid. The current quantizer shares it the same
    way.
    """

    def __init__(
        self, op: OpOverload, fused_op: OpOverload, config: QuantizationConfig
    ) -> None:
        super().__init__(
            op, fused_op, config, node_filter=lambda node: _pooled(node) is not None
        )

    def annotate_node(self, node: fx.Node) -> None:
        super().annotate_node(node)
        pooled = _pooled(node)
        assert pooled is not None
        pool, pool_input = pooled
        annotate_output_qspec(node, SharedQuantizationSpec((pool_input, pool)))


# Ops that move or pick out their input's values, unchanged.
_DATA_MOVEMENT = frozenset(
    {
        _aten.view.default,
        _aten.reshape.default,
        _aten._unsafe_view.default,
        _aten.permute.default,
        _aten.transpose.int,
        _aten.t.default,
        _aten.expand.default,
        _aten.unsqueeze.default,
        _aten.squeeze.default,
        _aten.squeeze.dim,
        _aten.squeeze.dims,
        _aten.flatten.using_ints,
        _aten.unflatten.int,
        _aten.clone.default,
        _aten.alias.default,
        _aten.slice.Tensor,
        _aten.select.int,
    }
)
# Their getitems pick out their input's values.
_SPLITS = frozenset(
    {
        _aten.split.Tensor,
        _aten.split_with_sizes.default,
        _aten.chunk.default,
        _aten.unbind.int,
    }
)


def _moved_from(node: fx.Node) -> fx.Node:
    """The node whose values node holds, through unannotated data movement."""
    while Q_ANNOTATION_KEY not in node.meta:
        data = node.args[0] if node.args else None
        if node.target is operator.getitem and isinstance(data, fx.Node):
            if data.target not in _SPLITS or Q_ANNOTATION_KEY in data.meta:
                break
            data = data.args[0]
        elif node.target not in _DATA_MOVEMENT:
            break
        if not isinstance(data, fx.Node):
            break
        node = data
    return node


def _output_qspec(node: fx.Node) -> Optional[QuantizationSpecBase]:
    """node's output qspec, through the qspecs it shares."""
    annotation = node.meta.get(Q_ANNOTATION_KEY)
    qspec = None if annotation is None else annotation.output_qspec
    while isinstance(qspec, SharedQuantizationSpec):
        shared = qspec.edge_or_node
        if isinstance(shared, fx.Node):
            annotation = shared.meta.get(Q_ANNOTATION_KEY)
            qspec = None if annotation is None else annotation.output_qspec
        else:
            arg, user = shared
            qspec = user.meta[Q_ANNOTATION_KEY].input_qspec_map.get(arg)
    return qspec


class _SharedReadsQuantizer(QuantizerBase):
    """An op reads another's output on that op's grid, through data movement.

    PT2E gives an op and its reader one observer only when they are adjacent
    and their qspecs agree. Through data movement, or from a fixed grid such
    as a softmax's, the reader gets a grid of its own, which a requantize then
    reaches. The two observers see the same values unless the op's grid is
    fixed or shared with its input, as a pool's is, so the current quantizer
    shares them, as this does. It amends the annotations of the quantizers
    ahead of it.
    """

    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        for node in model.graph.nodes:
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            if annotation is None:
                continue
            for arg, qspec in list(annotation.input_qspec_map.items()):
                if not isinstance(qspec, QuantizationSpec):
                    continue
                source = _moved_from(arg)
                output = _output_qspec(source)
                if output is None or getattr(output, "dtype", None) != qspec.dtype:
                    continue
                if source is not arg or isinstance(
                    output, FixedQParamsQuantizationSpec
                ):
                    annotation.input_qspec_map[arg] = SharedQuantizationSpec(source)
        return model

    def fuse(self, gm: fx.GraphModule) -> bool:
        return False

    def preserved_ops(self) -> Iterable[OpOverload]:
        return ()


def _constant_fill(node: object) -> Optional[float]:
    """The value a full or full_like node fills with."""
    if not isinstance(node, fx.Node) or node.target not in (
        _aten.full.default,
        _aten.full_like.default,
    ):
        return None
    value = node.args[1]
    return float(value) if isinstance(value, (int, float)) else None


def _clamp_tensor_to_scalars(model: fx.GraphModule) -> None:
    """A clamp to constant fills becomes a scalar clamp, which quantizes whole."""
    for node in model.graph.find_nodes(op="call_function", target=_aten.clamp.Tensor):
        inp = cast(fx.Node, get_arg(node, "input"))
        bounds: list[Argument] = [get_arg(node, "min"), get_arg(node, "max")]
        values = [_constant_fill(bound) for bound in bounds]
        if node.meta["val"].shape != inp.meta["val"].shape or any(
            bound is not None and value is None for bound, value in zip(bounds, values)
        ):
            continue
        node.target = _aten.clamp.default
        node.args = (inp, *values)
        node.kwargs = {}
        for bound in bounds:
            if isinstance(bound, fx.Node) and not bound.users:
                model.graph.erase_node(bound)


class _AnnotationRewrites(QuantizerBase):
    """Rewrites ahead of annotation that let more of the graph quantize whole.

    A scaled_dot_product_attention without a mask, dropout or GQA becomes
    bmm -> scale -> softmax -> bmm, as the current quantizer decomposes it,
    and the scale then folds into the qparams around it. Any other is
    kept whole, so ``to_edge`` decomposes it after quantization, into float
    ops as the current flow runs it: a mask's -inf would reach an observer.

    A clamp to constant fills becomes a scalar clamp.
    """

    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        return model

    def fuse(self, gm: fx.GraphModule) -> bool:
        return False

    def preserved_ops(self) -> Iterable[OpOverload]:
        return (_SDPA,)

    def transform_for_annotation(self, model: fx.GraphModule) -> fx.GraphModule:
        _clamp_tensor_to_scalars(model)
        attentions = model.graph.find_nodes(op="call_function", target=_SDPA)
        if attentions:
            for attention in attentions:
                _reshape_views_of(attention)
            for rewrite in (DecomposeSDPAPass(), MatmulToBmmPass()):
                model = cast(fx.GraphModule, rewrite.call(model).graph_module)
            for node in model.graph.find_nodes(
                op="call_function", target=_aten.softmax.int
            ):
                if len(node.args) == 2 and not node.kwargs:
                    node.target = _aten._softmax.default
                    node.args = (*node.args, False)
        model.recompile()
        return model


def _reshape_views_of(attention: fx.Node) -> None:
    """The views of an attention's output, through data movement, become
    reshapes.

    The attention may write its output transposed, as the flash kernels do,
    and the export elide a copy that would make a view after it valid. The
    decomposition writes it contiguously.
    """
    pending = list(attention.users)
    while pending:
        node = pending.pop()
        if node.target == _aten.view.default:
            node.target = _aten.reshape.default
        if node.target in _DATA_MOVEMENT:
            pending.extend(node.users)


def _is_spatial_mean(node: fx.Node) -> bool:
    """A mean over H and W of an NCHW tensor, which arm_avgpool_s8 can run."""
    dims: Argument = get_arg(node, "dim")
    return (
        get_arg(node, "input", fx.Node).meta["val"].dim() == 4
        and isinstance(dims, (list, tuple))
        and sorted(cast(int, d) % 4 for d in dims) == [2, 3]
        and get_arg(node, "dtype") is None
    )


def get_cortex_m_fused_quant_quantizer(
    *,
    per_channel_linear: bool = False,
    activation: QuantizationSpec = INT8_ACTIVATION_PER_TENSOR_QSPEC,
) -> FusedQuantQuantizer:
    """The quantizer for :func:`get_cortex_m_fused_quant_recipe`.

    The current Cortex-M quantizer's qspecs: int8 per-tensor activations and int8
    symmetric weights, per channel for convolutions and, with
    ``per_channel_linear``, for linears. Biases are int32 at the accumulator
    scale, and softmax outputs are on the fixed grid arm_softmax_s8 writes.
    ``activation`` replaces the activation qspec of every op but the
    convolutions, as the current quantizer's ``per_tensor_config`` does: with
    ``INT16_ACTIVATION_PER_TENSOR_QSPEC``, a division runs its int16 kernel.

    As in the current flow, an op reads its producer's grid through data
    movement, an activation after a max pool writes the pool's grid, an
    attention decomposes into bmm and softmax where it can, and a convolution
    or pool no kernel runs stays float.

    The quantizers ahead of the core ATen ones override them for the same op;
    the first quantizer to claim a node wins.
    """
    config = QuantizationConfig(
        activation, activation, INT8_WEIGHT_PER_TENSOR_QSPEC, None
    )
    weighted_config = QuantizationConfig(
        INT8_ACTIVATION_PER_TENSOR_QSPEC,
        INT8_ACTIVATION_PER_TENSOR_QSPEC,
        INT8_WEIGHT_PER_CHANNEL_QSPEC,
        None,
    )
    grouped_transposed_config = dataclasses.replace(
        weighted_config, weight=INT8_WEIGHT_PER_TENSOR_QSPEC
    )
    linear_config = (
        QuantizationConfig(activation, activation, INT8_WEIGHT_PER_CHANNEL_QSPEC, None)
        if per_channel_linear
        else config
    )
    softmax_config = QuantizationConfig(
        activation,
        # PT2E takes any spec here; torchao annotates the field more narrowly.
        # pyrefly: ignore[bad-argument-type]
        SOFTMAX_OUTPUT_FIXED_QSPEC,  # type: ignore[arg-type]
        None,
        None,
    )
    aten, fused_quant = torch.ops.aten, torch.ops.fused_quant

    def weighted(
        op: OpOverload, fused_op: OpOverload, op_config: QuantizationConfig
    ) -> QuantizerBase:
        return _QuantizedBiasOpQuantizer(
            op,
            fused_op,
            op_config,
            activation_names=("input",),
            weight_names=("weight",),
            other_names=("bias",),
        )

    def unary(name: str) -> QuantizerBase:
        return OpQuantizer(
            getattr(aten, name).default, getattr(fused_quant, name).default, config
        )

    return FusedQuantQuantizer(
        [
            _AnnotationRewrites(),
            *(
                _PooledActivationQuantizer(op, fused_op, config)
                for op, fused_op in (
                    (aten.relu.default, fused_quant.relu.default),
                    (aten.hardtanh.default, fused_quant.hard_tanh.default),
                    (aten.clamp.default, fused_quant.clamp.default),
                )
            ),
            # A convolution or pool no kernel runs stays float, as in the
            # current flow; quantized, it would still run in float.
            NoopQuantizer(
                aten.convolution.default,
                node_filter=lambda node: not _runs_on_cmsis_nn(node),
            ),
            NoopQuantizer(
                aten.max_pool2d_with_indices.default,
                node_filter=lambda node: not _max_pool_runs_on_cmsis_nn(node),
            ),
            NoopQuantizer(
                aten.avg_pool2d.default,
                node_filter=lambda node: get_arg(node, "divisor_override") is not None,
            ),
            _SharedQparamsQuantizer(
                aten.avg_pool2d.default, fused_quant.avg_pool2d.default, config
            ),
            _SharedQparamsQuantizer(
                aten.mean.dim,
                fused_quant.mean.dim,
                config,
                node_filter=_is_spatial_mean,
            ),
            _SharedQparamsQuantizer(
                aten.constant_pad_nd.default,
                fused_quant.constant_pad_nd.default,
                config,
            ),
            _SharedQparamsQuantizer(
                aten.minimum.default,
                fused_quant.minimum.default,
                config,
                activation_names=("input", "other"),
            ),
            _SharedQparamsQuantizer(
                aten.maximum.default,
                fused_quant.maximum.default,
                config,
                activation_names=("input", "other"),
            ),
            # prepare_pt2e folds a batch norm only into conv1d / conv2d, so they
            # are kept whole rather than decomposed to convolution.
            weighted(aten.conv1d.default, fused_quant.conv1d.default, weighted_config),
            weighted(aten.conv2d.default, fused_quant.conv2d.default, weighted_config),
            weighted(aten.linear.default, fused_quant.linear.default, linear_config),
            _QuantizedBiasConvolutionQuantizer(
                weighted_config=weighted_config,
                grouped_transposed_config=grouped_transposed_config,
            ),
            OpQuantizer(
                aten._softmax.default, fused_quant._softmax.default, softmax_config
            ),
            *get_core_aten_quantizers(config, weighted_config),
            unary("hardswish"),
            unary("hardsigmoid"),
            unary("silu"),
            *(unary(name) for name in _LUT_OPS),
            _SharedReadsQuantizer(),
        ]
    )
