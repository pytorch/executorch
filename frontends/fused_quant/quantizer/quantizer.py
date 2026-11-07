# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import re
from abc import ABC, abstractmethod
from typing import Callable, cast, Iterable, Optional

import torch
from executorch.backends.transforms.fused_quant.fuse_aten import (
    arg_names,
    fuse_aten,
    output_node,
)
from executorch.backends.transforms.fused_quant.graph_utils import is_dequantize_node
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.frontends.fused_quant.quantizer.qspecs import arg1_is_scalar
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverload
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.tools_common import stable_topological_sort
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    QuantizationConfig,
    Quantizer,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import (
    Q_ANNOTATION_KEY,
    QuantizationAnnotation,
)
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_input_qspec_map,
    annotate_output_qspec,
)


# Node meta flag set when a fused_quant quantizer claimed a node, so fuse() can
# tell its own claims from another PT2E quantizer's (which also set _annotated).
_OWNER_KEY = "fused_quant_owned"


class QuantizerBase(Quantizer, ABC):
    """A torchao Quantizer that also fuses."""

    def validate(self, model: fx.GraphModule) -> None:
        """Unused; satisfies the torchao Quantizer contract."""

    @abstractmethod
    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        """Stamp QuantizationAnnotation meta on the ops this quantizer owns."""
        ...

    @abstractmethod
    def fuse(self, gm: fx.GraphModule) -> bool:
        """Fold each dequant -> op -> quant into a fused op; return whether the
        graph changed."""
        ...

    @abstractmethod
    def preserved_ops(self) -> Iterable[OpOverload]:
        """Ops this quantizer matches, kept intact through decomposition."""
        ...


class OpQuantizer(QuantizerBase):
    """A single-op quantizer implementation: annotates and fuses a single op type.

    Args:
        op: The ATen op to match, annotate, and preserve from decomposition.
        fused_op: The fused_quant op to lower matched instances into.
        config: The per-instance quantization config.
        activation_names: Schema names of quantized activation inputs (annotated
            with config.input_activation).
        weight_names: Schema names of quantized weight inputs (annotated with
            config.weight).
        other_names: Schema names of tensor passthroughs (e.g. "bias") that
            occupy the fused op's tensor prefix with a null qparams block.
        output_indices: Outputs to quantize with config.output_activation. Defaults
            to just output 0; a multi-output op can pass e.g. (0, 1). All quantized
            outputs share the single config.output_activation.
        node_filter: Optional predicate restricting which op nodes this quantizer
            owns.
    """

    def __init__(
        self,
        op: OpOverload,
        fused_op: OpOverload,
        config: QuantizationConfig,
        *,
        activation_names: tuple[str, ...] = ("input",),
        weight_names: tuple[str, ...] = (),
        other_names: tuple[str, ...] = (),
        output_indices: tuple[int, ...] = (0,),
        node_filter: Optional[Callable[[fx.Node], bool]] = None,
    ) -> None:
        # Validate arg names.
        valid_names = arg_names(op)
        for name in (*activation_names, *weight_names, *other_names):
            if name not in valid_names:
                raise ValueError(
                    f"{type(self).__name__}: {name!r} is not an argument of {op}; "
                    f"expected one of {valid_names}"
                )

        self.op = op
        self.fused_op = fused_op
        self.config = config
        self.activation_names = activation_names
        self.weight_names = weight_names
        self.other_names = other_names
        self.output_indices = output_indices
        self.node_filter = node_filter

    def matches(self, node: fx.Node) -> bool:
        """Whether this quantizer handles node, the shared gate for annotate and fuse.

        Requires node to pass node_filter and every quantized edge to be a float tensor.
        """
        if self.node_filter is not None and not self.node_filter(node):
            return False

        def is_float_tensor(x: fx.Node) -> bool:
            # A dequantize node's output is float by definition; convert-inserted
            # dq/q nodes may not carry meta["val"], so check the target directly.
            if is_dequantize_node(x):
                return True
            return (
                isinstance(x.meta.get("val"), torch.Tensor)
                and x.meta["val"].is_floating_point()
            )

        for name in (*self.activation_names, *self.weight_names):
            # get_arg is not necessarily Node type, e.g. aten.add with scalar rhs.
            arg = get_arg(node, name)
            if not isinstance(arg, fx.Node) or not is_float_tensor(arg):
                return False
        if self.config.output_activation is not None:
            for output_index in self.output_indices:
                out = output_node(node, output_index)
                if out is not None and not is_float_tensor(out):
                    return False
        return True

    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        """Annotate every matching node in the graph. Skips already annotated nodes.

        Annotations are marked ``_annotated`` -- the flag other PT2E quantizers
        check before claiming a node -- so composing with them is first-claim-wins
        in either order.
        """
        for node in model.graph.find_nodes(op="call_function", target=self.op):
            if Q_ANNOTATION_KEY not in node.meta and self.matches(node):
                self.annotate_node(node)
                self._mark_owned(node)
        return model

    def _mark_owned(self, node: fx.Node) -> None:
        node.meta[_OWNER_KEY] = True
        outputs = [output_node(node, index) for index in self.output_indices]
        for annotated in (node, *outputs):
            if annotated is None:
                continue
            annotation = annotated.meta.get(Q_ANNOTATION_KEY)
            if annotation is not None:
                annotation._annotated = True

    def annotate_node(self, node: fx.Node) -> None:
        """Annotate node. Subclasses may override this for custom annotation."""
        for name in self.activation_names:
            annotate_input_qspec_map(
                node, cast(fx.Node, get_arg(node, name)), self.config.input_activation
            )
        for name in self.weight_names:
            annotate_input_qspec_map(
                node, cast(fx.Node, get_arg(node, name)), self.config.weight
            )
        # output_node is None if a multi-output op's getitem(output_index) is unused.
        if self.config.output_activation is not None:
            for output_index in self.output_indices:
                out = output_node(node, output_index)
                if out is not None:
                    annotate_output_qspec(out, self.config.output_activation)

    def fuse(self, gm: fx.GraphModule) -> bool:
        """Fold each annotated op node this quantizer matches into fused_op.

        A node is fused only when it matches and carries an annotation with a live
        qspec, so a node claimed by a no-op annotation or one merely sandwiched
        between its neighbors' dq/q is left alone. A node another PT2E quantizer
        claimed (``_annotated`` set, but not by a fused_quant quantizer) is never
        fused.
        """
        modified = False
        for node in list(gm.graph.find_nodes(op="call_function", target=self.op)):
            if not self.matches(node):
                continue
            ann = node.meta.get(Q_ANNOTATION_KEY)
            if ann is None or (ann._annotated and not node.meta.get(_OWNER_KEY)):
                continue
            if (
                not any(v is not None for v in ann.input_qspec_map.values())
                and ann.output_qspec is None
            ):
                continue
            self.fuse_node(gm, node)
            modified = True
        return modified

    def fuse_node(self, gm: fx.GraphModule, node: fx.Node) -> fx.Node:
        """Fuse a generic aten op to a matching fused_quant op.

        Subclasses may override for custom fusion logic.
        """
        return fuse_aten(
            node,
            self.fused_op,
            self.activation_names,
            self.weight_names,
            self.other_names,
            self.output_indices,
        )

    def preserved_ops(self) -> Iterable[OpOverload]:
        return (self.op,)


class NoopQuantizer(OpQuantizer):
    """Annotates matching ops with a no-op config so that later quantizers do not
    touch them; must be listed before the quantizer it suppresses.

    A skipped op stays an unfused float aten op, so the later to_edge decomposes
    it into core-aten; the backend must have a kernel for that or compilation
    fails at lowering.
    """

    def __init__(self, op: OpOverload, node_filter: Callable[[fx.Node], bool]) -> None:
        super().__init__(
            op,
            op,  # fused_op target, unused.
            QuantizationConfig(None, None, None, None),
            activation_names=(),
            node_filter=node_filter,
        )

    def annotate_node(self, node: fx.Node) -> None:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(input_qspec_map={})

    def fuse(self, gm: fx.GraphModule) -> bool:
        return False


class MaxPoolQuantizer(OpQuantizer):
    """max_pool2d_with_indices: output 0 (the pooled values, via getitem 0) shares
    the input's observer; the indices output is left unquantized."""

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.fused_quant.max_pool2d_with_indices.default,
            config,
        )

    def annotate_node(self, node: fx.Node) -> None:
        """Annotate as usual, then re-point output 0 to share input 0's observer."""
        super().annotate_node(node)
        out = output_node(node)
        if out is not None:
            annotate_output_qspec(
                out,
                SharedQuantizationSpec((cast(fx.Node, node.args[0]), node)),
            )


class FusedQuantQuantizer(ComposableQuantizer, QuantizerBase):
    """Composes several QuantizerBase instances into a single Quantizer.

    Order matters: annotation runs in list order and the first quantizer to claim a
    node wins (later ones skip an already-annotated node). A quantizer that narrows
    or suppresses another, e.g. a node_filtered override or a NoopQuantizer, must
    come before the general one it overrides.
    """

    def __init__(self, quantizers: list[QuantizerBase]) -> None:
        for q in quantizers:
            if not isinstance(q, QuantizerBase):
                raise TypeError(f"{type(q).__name__} is not a QuantizerBase")
        super().__init__(list(quantizers))

    @property
    def child_quantizers(self) -> list[QuantizerBase]:
        """The composed quantizers, typed as the QuantizerBase instances __init__
        admits (ComposableQuantizer types them as plain Quantizer)."""
        return cast(list[QuantizerBase], self.quantizers)

    def fuse(self, gm: fx.GraphModule) -> bool:
        """Fuse via every child quantizer; return whether anything fused.

        ``gm`` is the output of ``convert_pt2e``, which is first cleaned up so
        the children can match on it; see :func:`_repair_convert_output`.
        """
        _repair_convert_output(gm)
        modified = False
        for q in self.child_quantizers:
            modified |= q.fuse(gm)
        if modified:
            stable_topological_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.recompile()
        return modified

    def preserved_ops(self) -> list[OpOverload]:
        """The union of every child quantizer's preserved ops."""
        return [op for q in self.child_quantizers for op in q.preserved_ops()]


def _repair_convert_output(gm: fx.GraphModule) -> None:
    """Work around two ``convert_pt2e`` (torchao) gaps that fusion trips on.

    ``AffineQuantizedObserverBase.convert`` inserts ``quantize_affine`` /
    ``dequantize_affine`` as the ``OpOverloadPacket`` rather than ``.default``,
    which fusion does not match. And ``convert_pt2e`` rewrites the graph without
    refreshing ``meta["val"]``, which fusion reads for shapes and dtypes.
    """
    replacements = {
        torch.ops.torchao.quantize_affine: torch.ops.torchao.quantize_affine.default,
        torch.ops.torchao.dequantize_affine: torch.ops.torchao.dequantize_affine.default,
    }
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in replacements:
            node.target = replacements[node.target]
    gm.recompile()

    # A hand-built graph may carry no fake values at all; there is nothing to refresh.
    placeholders = gm.graph.find_nodes(op="placeholder")
    fake_mode = _detect_fake_mode_from_gm(gm)
    if fake_mode is None or any("val" not in node.meta for node in placeholders):
        return
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
        *(node.meta["val"] for node in placeholders)
    )


# Why add/mul/sub are each registered twice -- once for tensor operands and once
# for a scalar operand:
#
# PyTorch does NOT give scalar arithmetic its own op overload. `tensor + 1e-5`
# (e.g. RMSNorm's `variance + eps`) traces to `aten.add.Tensor(tensor, 1e-5)` --
# the `.Tensor` overload carrying a Python scalar literal in arg 1, NOT a
# separate `aten.add.Scalar`. So the op type (`aten.add.Tensor`) alone does not
# tell us whether arg 1 is a tensor or a scalar; both land on the same op.
#
# Annotating a scalar arg as a quant edge breaks PT2E prepare, so the two operand
# shapes are handled separately:
#   - activation_names=("input", "other") quantizes both tensor operands; the
#     float-tensor guard makes it skip a scalar arg 1.
#   - activation_names=("input",) + node_filter=arg1_is_scalar quantizes only the
#     tensor input and leaves the scalar unquantized.
# Between them every add/mul/sub is covered. The scalar variant folds into
# `fused_quant.{add,mul,sub}.Scalar`, which pass the scalar through as a trailing
# arg.
def _get_binary_quantizers(config: QuantizationConfig) -> list[QuantizerBase]:
    quantizers: list[QuantizerBase] = []
    for aten_op, tensor_op, scalar_op in (
        (
            torch.ops.aten.add.Tensor,
            torch.ops.fused_quant.add.default,
            torch.ops.fused_quant.add.Scalar,
        ),
        (
            torch.ops.aten.mul.Tensor,
            torch.ops.fused_quant.mul.default,
            torch.ops.fused_quant.mul.Scalar,
        ),
        (
            torch.ops.aten.sub.Tensor,
            torch.ops.fused_quant.sub.default,
            torch.ops.fused_quant.sub.Scalar,
        ),
    ):
        quantizers.append(
            OpQuantizer(aten_op, tensor_op, config, activation_names=("input", "other"))
        )
        quantizers.append(
            OpQuantizer(
                aten_op,
                scalar_op,
                config,
                activation_names=("input",),
                node_filter=arg1_is_scalar,
            )
        )
    return quantizers


def _transposed_convolution_quantizers(
    config: QuantizationConfig,
    weighted_config: QuantizationConfig,
) -> list[QuantizerBase]:
    """Quantizers for ``aten.convolution`` with ``transposed=True``.

    A transposed convolution's weight is ``[C_in, C_out // groups, ...]``, so
    per-output-channel weight scales sit on axis 1, not axis 0. With groups > 1
    axis 1 holds only ``C_out // groups`` channels, so those fall back to the
    per-tensor weights of ``config``. Listed before the general
    convolution quantizer so they claim transposed nodes first.
    """
    weight = weighted_config.weight
    if weight is None or weight.qscheme not in (
        torch.per_channel_affine,
        torch.per_channel_symmetric,
    ):
        return []
    axis_1 = dataclasses.replace(
        weighted_config, weight=dataclasses.replace(weight, ch_axis=1)
    )

    def transposed(single_group: bool):
        def node_filter(node: fx.Node) -> bool:
            return (
                bool(get_arg(node, "transposed"))
                and (get_arg(node, "groups") == 1) == single_group
            )

        return node_filter

    return [
        OpQuantizer(
            torch.ops.aten.convolution.default,
            torch.ops.fused_quant.convolution.default,
            conv_config,
            activation_names=("input",),
            weight_names=("weight",),
            other_names=("bias",),
            node_filter=transposed(single_group),
        )
        for conv_config, single_group in ((axis_1, True), (config, False))
    ]


def get_core_aten_quantizers(
    config: QuantizationConfig,
    weighted_config: QuantizationConfig,
) -> list[QuantizerBase]:
    """Quantizers for every core ATen op that has a ``fused_quant`` equivalent.

    Core ATen ops survive export's default decompositions, so none of these
    depends on its op being preserved. Backends add quantizers for any non-core
    op they keep whole.

    Convolution uses ``weighted_config``; everything else uses ``config``. Both
    are ordinary ``QuantizationConfig`` values -- the split is by which ops they
    apply to, not by granularity, and nothing here requires either one to be
    per-tensor or per-channel.
    """
    unary_ops = (
        (torch.ops.aten.sigmoid.default, torch.ops.fused_quant.sigmoid.default),
        (torch.ops.aten.tanh.default, torch.ops.fused_quant.tanh.default),
        (torch.ops.aten.hardtanh.default, torch.ops.fused_quant.hard_tanh.default),
        (torch.ops.aten.gelu.default, torch.ops.fused_quant.gelu.default),
        (
            torch.ops.aten.native_layer_norm.default,
            torch.ops.fused_quant.native_layer_norm.default,
        ),
    )
    return [
        *_get_binary_quantizers(config),
        OpQuantizer(
            torch.ops.aten.bmm.default,
            torch.ops.fused_quant.bmm.default,
            config,
            activation_names=("input", "mat2"),
        ),
        OpQuantizer(
            torch.ops.aten.relu.default,
            torch.ops.fused_quant.relu.default,
            config,
        ),
        MaxPoolQuantizer(config),
        OpQuantizer(
            torch.ops.aten.avg_pool2d.default,
            torch.ops.fused_quant.avg_pool2d.default,
            config,
        ),
        *_transposed_convolution_quantizers(config, weighted_config),
        OpQuantizer(
            torch.ops.aten.convolution.default,
            torch.ops.fused_quant.convolution.default,
            weighted_config,
            activation_names=("input",),
            weight_names=("weight",),
            other_names=("bias",),
        ),
        *(OpQuantizer(aten_op, fused_op, config) for aten_op, fused_op in unary_ops),
    ]


# =============================================================================
# node_filter predicates
# =============================================================================


def module_fqn_filter(pattern: str) -> Callable[[fx.Node], bool]:
    """A node_filter matching nodes whose module FQN matches pattern (regex
    search over nn_module_stack)."""
    regex = re.compile(pattern)

    def _filter(node: fx.Node) -> bool:
        return any(
            regex.search(fqn) is not None
            for fqn, _ in node.meta.get("nn_module_stack", {}).values()
        )

    return _filter
