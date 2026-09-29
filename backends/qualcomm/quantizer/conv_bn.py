# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Partition-level annotation for conv + batchnorm under QAT.

The per-node annotators in ``annotators/`` treat conv and batchnorm as unrelated
ops. That is correct for PTQ, where batchnorm is folded into the conv before the
graph is quantized, but it breaks QAT.

``prepare_qat_pt2e`` annotates the graph *before* torchao's ``_fuse_conv_bn_qat``
rewrites each conv+bn into the QAT folding pattern::

    scale = bn_weight / sqrt(bn_running_var + eps)
    y     = conv(x, w * scale.reshape(w_shape)) / scale.reshape(b_shape)
    out   = batch_norm(y, ...)

torchao does that ordering deliberately -- "Perform fusion after annotate to
avoid quantizing ops in the new subgraph" -- but the rewrite copies each matched
node's ``meta`` onto its replacement and re-points ``SharedQuantizationSpec``
chains at the new nodes, so a qspec on the *conv output* ends up on the scale
arithmetic. ``convert_pt2e``'s ``_fold_conv_bn_qat`` matches a pattern carrying
q/dq at exactly three points -- conv input, scaled weight, bn output -- so the
extra q/dq makes the subgraph matcher miss and batchnorm is never folded. Float
batchnorm then reaches the QNN partitioner, which has no node visitor for the
training variant, drops it along with its FP16 scale arithmetic, and shatters
the graph into one partition per conv. A partition left with only constant
inputs fails context-binary serialization with the opaque
``No graph inputs present for graph [0]`` / ``0x7532``.

Annotating the partition as a unit -- qspecs on the conv's inputs, the output
qspec on the *last* node of the partition, nothing on the conv output -- gives
the fold the shape it expects. This mirrors ``_do_annotate_conv_bn`` in the
XNNPACK quantizer and the ``Conv*Bn*Quantizer`` family in BoltNN; QNN was the
only one of the three without it.
"""

import operator
from typing import Callable, List, Optional, Set

import torch
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

from .qconfig import QuantizationConfig
from .rules import _is_annotated, _mark_nodes_as_annotated, Q_ANNOTATION_KEY

# Only the overloads torchao's `_fuse_conv_bn_qat` / `_fold_conv_bn_qat` can
# actually match. Both trace their patterns from `F.conv1d`, `F.conv2d`,
# `F.conv_transpose1d` and `F.conv_transpose2d`, which export to exactly these
# four targets.
#
# Claiming a conv the fold cannot match is worse than not claiming it: this pass
# drops the conv's output qspec on the assumption the fold will move the bn
# output observer onto it, so if the fold misses, the conv is left emitting
# float32 into a quantized graph and QNN rejects it (`0x232 != 0x408`, FLOAT_32
# meeting UFIXED_POINT_8). `aten.conv2d.padding` -- produced by
# `nn.Conv2d(..., padding="same")` -- hit exactly that. `aten.convolution.default`
# only appears after decomposition, well after this pass runs, so it never
# matched anything here either.
#
# Note `_is_conv_or_conv_transpose_node` is NOT a usable guard: it returns True
# for `conv2d.padding`.
CONV_TARGETS = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv_transpose1d.default,
    torch.ops.aten.conv_transpose2d.input,
)

# ``batch_norm.default`` returns a bare tensor; every other variant returns a
# tuple whose element 0 is the normalized output.
SINGLE_OUTPUT_BN_TARGETS = (torch.ops.aten.batch_norm.default,)

BN_TARGETS = SINGLE_OUTPUT_BN_TARGETS + (
    torch.ops.aten._native_batch_norm_legit.default,
    torch.ops.aten._native_batch_norm_legit_functional.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    torch.ops.aten.cudnn_batch_norm.default,
)


def _sole_user(node: Node) -> Optional[Node]:
    return next(iter(node.users)) if len(node.users) == 1 else None


def _bn_output(bn: Node) -> Optional[Node]:
    """The node carrying the batchnorm's normalized output."""
    if bn.target in SINGLE_OUTPUT_BN_TARGETS:
        return bn
    for user in bn.users:
        if user.target is operator.getitem and user.args[1] == 0:
            return user
    return None


def _materialize_conv_bias(gm: GraphModule, conv: Node) -> Optional[Node]:
    """Give a bias-less conv a zero bias so the fold produces a *quantized* one.

    When the conv has no bias, ``fold_bn_weights_into_conv_node`` materializes the
    folded bias as a fresh fp32 parameter and explicitly leaves it unquantized
    ("NOTE: here we assume the bias of conv is not quantized!"). QNN rejects a
    per-channel-quantized conv carrying a float bias, so the conv drops out of the
    partition and its orphaned weight dequant has to become a standalone
    Dequantize -- which QNN does not support per-channel.

    Adding a zero bias is a numerical no-op that gives the annotator a node to
    attach the derived bias qspec to. The fold then takes its
    ``conv_bias_node is not None`` branch and writes the folded values into that
    already-quantized node instead of inventing an unquantized one.
    """
    args = list(conv.args)
    while len(args) < 3:
        args.append(None)
    if isinstance(args[2], Node):
        return args[2]

    val = conv.meta.get("val")
    if val is None:
        return None

    zeros = torch.zeros(val.shape[1], dtype=val.dtype, device=val.device)
    name = f"{conv.name}_zero_bias"
    gm.register_parameter(name, torch.nn.Parameter(zeros, requires_grad=False))
    with gm.graph.inserting_before(conv):
        bias = gm.graph.get_attr(name)
    bias.meta["val"] = val.fake_mode.from_tensor(zeros, static_shapes=True)

    args[2] = bias
    conv.args = tuple(args)
    return bias


def _annotate_partition(
    gm: GraphModule,
    conv: Node,
    partition: List[Node],
    output_node: Node,
    quantization_config: QuantizationConfig,
) -> None:
    input_qspec_map = {}

    act = conv.args[0]
    if isinstance(act, Node) and quantization_config.input_activation is not None:
        input_qspec_map[act] = quantization_config.input_activation

    weight = conv.args[1]
    assert isinstance(weight, Node)
    input_qspec_map[weight] = quantization_config.weight

    bias = _materialize_conv_bias(gm, conv)
    if bias is not None and quantization_config.bias is not None:
        input_qspec_map[bias] = (
            quantization_config.bias(conv)
            if callable(quantization_config.bias)
            else quantization_config.bias
        )

    # Deliberately no output_qspec on the conv -- an observer there lands inside
    # the QAT fusion and stops `_fold_conv_bn_qat` from matching.
    conv.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        _annotated=True,
    )
    _mark_nodes_as_annotated(partition)
    output_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )


def annotate_conv_bn_partitions(
    gm: GraphModule,
    get_quant_config: Callable[[Node], Optional[QuantizationConfig]],
    discard_nodes: Set[str],
) -> int:
    """Annotate every conv+bn chain as a single partition.

    Must run before the per-node annotation pass; the nodes it claims are marked
    annotated so the per-node annotators skip them.

    A trailing relu is deliberately NOT absorbed into the partition. XNNPACK's
    equivalent puts the output qspec on the relu because it fuses conv+relu into
    one backend op, but QNN lowers conv and relu to separate ops and rejects a
    conv whose output is unquantized (`0x232 != 0x408`, FLOAT_32 meeting
    UFIXED_POINT_8). Keeping the qspec on the bn output -- which the fold
    transfers onto the conv output -- leaves the relu to the per-node annotator,
    which gives it its own quantized input and output.

    Returns the number of partitions annotated.
    """
    count = 0
    for conv in list(gm.graph.nodes):
        if conv.op != "call_function" or conv.target not in CONV_TARGETS:
            continue
        if conv.name in discard_nodes:
            continue

        bn = _sole_user(conv)
        if bn is None or bn.target not in BN_TARGETS:
            continue
        bn_out = _bn_output(bn)
        if bn_out is None:
            continue

        partition = [conv, bn] if bn_out is bn else [conv, bn, bn_out]
        output_node = bn_out

        if _is_annotated(partition):
            continue
        quantization_config = get_quant_config(conv)
        if quantization_config is None:
            continue

        _annotate_partition(gm, conv, partition, output_node, quantization_config)
        count += 1

    if count:
        gm.recompile()
    return count
