# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
from abc import ABC, abstractmethod
from typing import Callable, cast, Iterable, Optional

import torch
from executorch.backends.fused_quant.fuse_aten import arg_names, fuse_aten, output_node
from executorch.backends.fused_quant.graph_utils import is_dequantize_node
from executorch.backends.fused_quant.pre_quantize_passes.replace_mm_with_bmm import (
    ReplaceMmWithBmm,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from torch import fx
from torch._ops import OpOverload
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
        """Annotate every matching node in the graph. Skips already annotated nodes."""
        for node in model.graph.find_nodes(op="call_function", target=self.op):
            if (
                node.meta.get("source_fn_stack") is not None
                and Q_ANNOTATION_KEY not in node.meta
                and self.matches(node)
            ):
                self.annotate_node(node)
        return model

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
        """Fold each owned, annotated op node into fused_op.

        A node is fused only when it matches, carries an annotation, and that annotation
        has a live qspec, so a node claimed by a no-op annotation or one merely
        sandwiched between its neighbors' dq/q is left alone.
        """
        modified = False
        for node in list(gm.graph.find_nodes(op="call_function", target=self.op)):
            if not self.matches(node):
                continue
            ann = node.meta.get(Q_ANNOTATION_KEY)
            if ann is None:
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


class BmmQuantizer(OpQuantizer):
    """Quantize ``aten.bmm``, first rewriting every ``aten.mm`` into a unit-batch bmm.

    ``mm`` has no ``fused_quant`` equivalent, so left alone it stays an unfused
    float op on the DSP. Rewriting it in the pre-annotation transform puts the
    matmul on the fused bmm path -- the new bmm nodes are then annotated like any
    other -- and lets Turing TCE delegation pick it up.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(
            torch.ops.aten.bmm.default,
            torch.ops.fused_quant.bmm.default,
            config,
            activation_names=("input", "mat2"),
        )

    def transform_for_annotation(self, model: fx.GraphModule) -> fx.GraphModule:
        ReplaceMmWithBmm().call(model)
        return model


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

    def fuse(self, gm: fx.GraphModule) -> bool:
        """Fuse via every child quantizer; return whether anything fused."""
        modified = False
        for q in self.quantizers:
            modified |= cast(QuantizerBase, q).fuse(gm)
        if modified:
            stable_topological_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.recompile()
        return modified

    def preserved_ops(self) -> list[OpOverload]:
        """The union of every child quantizer's preserved ops."""
        return [
            op for q in self.quantizers for op in cast(QuantizerBase, q).preserved_ops()
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
