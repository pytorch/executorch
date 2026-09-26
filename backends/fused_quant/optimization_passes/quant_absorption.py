# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import operator
from typing import Optional

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_qparams_from_node,
    get_scale,
    get_zero_point,
    split_fused_arg_names,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


logger: logging.Logger = logging.getLogger(__name__)

_DEQUANT_PER_TENSOR: EdgeOpOverload = (
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
)
_QUANT_PER_TENSOR: EdgeOpOverload = (
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
)


def _is_fused_quant_op(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, EdgeOpOverload)
        and node.target.namespace == "fused_quant"
    )


def _has_per_tensor_out_qparams(node: fx.Node) -> bool:
    """Check if a fused_quant op has per-tensor (non-None) output quantization.
    Note: This is just to simplify this pass since this is the most common case,
    but it is possible that the output uses per-channel quantization, and we can
    still fuse downstream dq/quant ops into the fused_quant op.
    """
    out_scale = get_arg(node, "out_scale", Optional[fx.Node])
    if not isinstance(out_scale, fx.Node):
        return False
    # Per-tensor output is a singleton scale (granularity is encoded by shape).
    return out_scale.meta["val"].numel() == 1


def _has_unquantized_out(node: fx.Node) -> bool:
    """Check if a fused_quant op leaves its output unquantized (out_scale is None).

    Most fused ops fold their output quantize at fusion time, but some (notably
    fused_quant.embedding) deliberately keep a null output block -- their result
    is float and a downstream consumer quantizes it.
    """
    return get_arg(node, "out_scale", Optional[fx.Node]) is None


# Pure data-movement ops that are transparent to the requantization being
# absorbed. This is only safe because the pass requires *per-tensor* input or
# output quantization: a scalar scale/zero_point applies uniformly to every
# element regardless of layout, so reshapes and permutes leave it untouched. We
# would need to be more careful if we wanted to support fusion of different
# types of quantization.
_PASSTHROUGH_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.view_copy.default,
}


def _find_dequant_through_passthrough(
    node: fx.Node,
) -> tuple[fx.Node | None, list[fx.Node]]:
    """Walk single-user passthrough ops from node to find a dequant.

    Returns (dequant_node, passthrough_chain) if found, else (None, []).
    The passthrough_chain is the ordered list of passthrough nodes between the
    fused op and the dequant (node is the first element, the immediate
    predecessor of the dequant is the last). After absorption the quant's users
    consume passthrough_chain[-1], and every node in the chain carries the
    fused op's requantized output, so their meta['val'] dtype must follow it.
    """
    chain: list[fx.Node] = []
    cursor = node
    while len(cursor.users) == 1:
        child = next(iter(cursor.users.keys()))
        chain.append(cursor)
        if child.target == _DEQUANT_PER_TENSOR:
            return child, chain
        if child.target not in _PASSTHROUGH_TARGETS:
            return None, []
        cursor = child
    return None, []


def _find_quant_through_input_passthrough(
    node: fx.Node,
) -> tuple[fx.Node | None, list[fx.Node]]:
    """Walk backward through single-user passthrough ops to find a quantize.

    ``node`` is a fused consumer's input. The returned chain is ordered from the
    consumer toward the quantize, so ``chain[-1]`` is the passthrough whose input
    must be rewired when the quantize is absorbed.
    """
    chain: list[fx.Node] = []
    cursor = node
    while cursor.target in _PASSTHROUGH_TARGETS:
        if len(cursor.users) != 1:
            return None, []
        chain.append(cursor)
        predecessor = cursor.args[0]
        if not isinstance(predecessor, fx.Node):
            return None, []
        cursor = predecessor
    return (cursor, chain) if cursor.target == _QUANT_PER_TENSOR else (None, [])


def _output_zero_node(node: fx.Node) -> fx.Node | None:
    """The node carrying the fused op's output-0 value.

    Single-output fused ops produce their value directly, so it is ``node``
    itself. Multi-output ops (e.g. ``fused_quant.native_layer_norm`` returning
    ``(out, mean, rstd)``) expose each output through a ``getitem`` accessor;
    only output 0 is quantized (its qparams live in the op's ``out_scale`` /
    ``out_zero_point`` block), so its value flows through ``getitem(node, 0)``.
    Returns that accessor for a multi-output op, ``node`` for a single-output op,
    or None when a multi-output op's output 0 isn't a single clean getitem
    (nothing to absorb through).
    """
    if not isinstance(node.meta.get("val"), (tuple, list)):
        return node
    getitems = [
        u for u in node.users if u.target == operator.getitem and u.args[1] == 0
    ]
    return getitems[0] if len(getitems) == 1 else None


def _compose_output_qparams(
    exported_program: ExportedProgram,
    producer: fx.Node,
    dequantize: fx.Node,
    quantize: fx.Node,
) -> tuple[float, int] | None:
    """Compose Q(producer) → DQ → Q into one zero-error affine quantizer.

    Let ``x`` be the producer's floating-point result, ``q_A`` its original
    quantized output, and ``y`` the value after the possibly mismatched
    dequantize:

      q_A = x / s_A + z_A
      y = (q_A - z_D) * s_D
      q_B = y / s_B + z_B

    Ignoring rounding and saturation, choosing

      s_R = s_A * s_B / s_D
      z_R = z_B + (z_A - z_D) * s_D / s_B

    makes the replacement producer output ``q_R = Q(R, x)`` equal ``q_B``.
    A standard affine quantizer can represent this composition only when
    ``z_R`` is integral.

    R includes the effects of A, D, and B. If the dequantize has surviving
    float users, DQ(B) undoes just the B part and recovers ``y``.
    """
    producer_qparams = get_qparams_from_node(producer, "out")
    assert producer_qparams is not None and producer_qparams.is_per_tensor()
    if (
        producer_qparams.dtype != get_arg(dequantize, "dtype", torch.dtype)
        or producer_qparams.quant_min != get_arg(dequantize, "quant_min", int)
        or producer_qparams.quant_max != get_arg(dequantize, "quant_max", int)
    ):
        return None

    producer_scale = get_scale(exported_program, producer_qparams)
    dequant_scale = get_arg(dequantize, "scale", float)
    output_scale = get_arg(quantize, "scale", float)
    if any(
        not math.isfinite(scale) or scale <= 0
        for scale in (producer_scale, dequant_scale, output_scale)
    ):
        return None

    producer_zp = get_zero_point(exported_program, producer_qparams)
    dequant_zp = get_arg(dequantize, "zero_point", int)
    output_zp = get_arg(quantize, "zero_point", int)
    replacement_scale = producer_scale * output_scale / dequant_scale
    replacement_zp_float = (
        output_zp + (producer_zp - dequant_zp) * dequant_scale / output_scale
    )
    replacement_zp = round(replacement_zp_float)
    if (
        not math.isfinite(replacement_scale)
        or replacement_scale <= 0
        or not math.isclose(
            replacement_zp_float, replacement_zp, rel_tol=0.0, abs_tol=1e-6
        )
        or not get_arg(quantize, "quant_min", int)
        <= replacement_zp
        <= get_arg(quantize, "quant_max", int)
    ):
        return None
    return replacement_scale, replacement_zp


def _compose_input_qparams(
    exported_program: ExportedProgram,
    dequantize: fx.Node,
    quantize: fx.Node,
    consumer: fx.Node,
    input_name: str,
) -> tuple[float, int] | None:
    """Compose DQ(D) → Q(B) → consumer DQ(C) into consumer DQ(R).

    Ignoring rounding and saturation, choosing

      s_R = s_D * s_C / s_B
      z_R = z_D + (z_C - z_B) * s_B / s_D

    preserves the value seen by the consumer. A standard affine quantizer can
    represent the composition only when ``z_R`` is integral.
    """
    consumer_qparams = get_qparams_from_node(consumer, input_name)
    assert consumer_qparams is not None and consumer_qparams.is_per_tensor()
    if (
        get_arg(quantize, "quant_min", int) != consumer_qparams.quant_min
        or get_arg(quantize, "quant_max", int) != consumer_qparams.quant_max
    ):
        return None

    dequant_scale = get_arg(dequantize, "scale", float)
    quant_scale = get_arg(quantize, "scale", float)
    consumer_scale = get_scale(exported_program, consumer_qparams)
    if any(
        not math.isfinite(scale) or scale <= 0
        for scale in (dequant_scale, quant_scale, consumer_scale)
    ):
        return None

    dequant_zp = get_arg(dequantize, "zero_point", int)
    quant_zp = get_arg(quantize, "zero_point", int)
    consumer_zp = get_zero_point(exported_program, consumer_qparams)
    replacement_scale = dequant_scale * consumer_scale / quant_scale
    replacement_zp_float = (
        dequant_zp + (consumer_zp - quant_zp) * quant_scale / dequant_scale
    )
    replacement_zp = round(replacement_zp_float)
    if (
        not math.isfinite(replacement_scale)
        or replacement_scale <= 0
        or not math.isclose(
            replacement_zp_float, replacement_zp, rel_tol=0.0, abs_tol=1e-6
        )
        or not get_arg(dequantize, "quant_min", int)
        <= replacement_zp
        <= get_arg(dequantize, "quant_max", int)
    ):
        return None
    return replacement_scale, replacement_zp


def _try_absorb_output(
    ep: ExportedProgram, node: fx.Node, graph: fx.Graph, absorb_with_fork: bool
) -> bool:
    """Try to absorb a downstream dequant→quant into node's out qparams.

    Pattern 1 (single user):
      fused_quant Q(A) → [getitem 0] → [passthrough]* → DQ(D) → Q(B) → ...
      Result: fused_quant Q(compose(A, D, B)) → [getitem 0] → [passthrough]* → ...

    Pattern 2 (forked dequant, requires absorb_with_fork=True):
      fused_quant Q(A) → [getitem 0] → [passthrough]* → DQ(D) ┬→ Q(B) → ...
                                                              └→ other float users

      Result: fused_quant Q(R) → [getitem 0] → [passthrough]* ┬→ former Q(B) users
                                                              └→ DQ(B) → other float users
              where R = compose(A, D, B)

    When A == D, compose(A, D, B) == B, which is the ordinary removal of an
    identity quantize/dequantize boundary. A mismatch between A and D encodes
    a real affine operation and must be retained in the composed qparams.

    In the forked case, R includes A, D, and B. The former quantized users
    consume the composed integer output directly. DQ(B) on the float branch
    undoes just the B part, leaving the original A-to-D effect intact. DQ(R)
    would undo the entire composition and incorrectly erase that effect. When
    A == D, R == B, so the distinction is invisible.

    ``out_node`` is the node carrying output 0: the fused op itself for a
    single-output op, or ``getitem(node, 0)`` for a multi-output op (only output 0
    is quantized, and its qparams are the op's ``out_scale`` block). The walk to
    the dequant, and all retyping, run from ``out_node`` -- the op node itself is
    left alone for a multi-output op (its ``meta['val']`` is the output tuple).

    [passthrough]* is an optional chain of single-user data-movement ops
    (e.g. permute_copy) that are transparent to quantization parameters. The
    chain's structure is unaffected, but because absorbing the quant can change
    the output dtype (e.g. int8 → uint8), each passthrough node's meta['val'] is
    retyped to the absorbed dtype.
    """
    out_node = _output_zero_node(node)
    if out_node is None or len(out_node.users) != 1:
        return False

    user = next(iter(out_node.users.keys()))

    # passthrough_chain is the (possibly empty) list of data-movement ops
    # between the fused op and the dequant. Every node in it carries the fused
    # op's quantized output, so its dtype must follow the new output qparams.
    passthrough_chain: list[fx.Node] = []

    # Direct case: fused_quant [→ getitem 0] → dequant
    if user.target == _DEQUANT_PER_TENSOR:
        dequant_node = user
        # The node whose quantized output replaces the quant
        quantized_source = out_node
    elif user.target in _PASSTHROUGH_TARGETS:
        # Passthrough case: walk through passthrough ops to find dequant
        dequant_node, passthrough_chain = _find_dequant_through_passthrough(user)
        if dequant_node is None:
            return False
        quantized_source = passthrough_chain[-1]
    else:
        return False

    quant_nodes = [u for u in dequant_node.users if u.target == _QUANT_PER_TENSOR]
    if len(quant_nodes) != 1:
        return False
    quant_node = quant_nodes[0]

    has_other_users = len(dequant_node.users) > 1

    if has_other_users and not absorb_with_fork:
        return False

    replacement_qparams = _compose_output_qparams(ep, node, dequant_node, quant_node)
    if replacement_qparams is None:
        return False
    replacement_scale, replacement_zp = replacement_qparams

    downstream_scale = get_arg(quant_node, "scale", float)
    downstream_zp = get_arg(quant_node, "zero_point", int)
    # The downstream quant may target a different dtype than the fused op's
    # original output (e.g. int8 → uint8). Absorbing it means the fused op now
    # produces that dtype, so it must propagate to the fused op's out qparams
    # and to every node carrying its output (the passthrough chain, and the
    # kept dequant's input in the forked case).
    new_dtype = get_arg(quant_node, "dtype", torch.dtype)
    new_quant_min = get_arg(quant_node, "quant_min", int)
    new_quant_max = get_arg(quant_node, "quant_max", int)

    _set_qparams(
        ep,
        node,
        "out",
        scale=replacement_scale,
        zero_point=replacement_zp,
        dtype=new_dtype,
        quant_min=new_quant_min,
        quant_max=new_quant_max,
    )

    quant_node.replace_all_uses_with(quantized_source)
    graph.erase_node(quant_node)

    if has_other_users:
        # R includes A, D, and B. DQ(B) undoes just the B part for float users.
        set_arg(dequant_node, "scale", downstream_scale)
        set_arg(dequant_node, "zero_point", downstream_zp)
        set_arg(dequant_node, "dtype", new_dtype)
        set_arg(dequant_node, "quant_min", new_quant_min)
        set_arg(dequant_node, "quant_max", new_quant_max)
    else:
        graph.erase_node(dequant_node)

    # out_node (the fused op for single-output, or its getitem-0 for
    # multi-output) and every passthrough node now carry the requantized value
    # flowing into the quant's former users, so retype their meta['val']. The
    # multi-output op node itself keeps its output-tuple val untouched.
    for retyped in (out_node, *passthrough_chain):
        val = retyped.meta.get("val")
        if isinstance(val, torch.Tensor):
            retyped.meta["val"] = val.to(new_dtype)

    return True


def _try_absorb_quant_into_unquantized_out(
    exported_program: ExportedProgram, node: fx.Node, graph: fx.Graph
) -> bool:
    """Absorb a bare quantize that follows a fused op whose output is unquantized.

    Pattern:
      fused_quant(float out) → quant(B) → ...
      Result: fused_quant(out qparams = B) → ...

    Some fused ops (notably fused_quant.embedding) leave their output unquantized
    at fusion time -- the produced value is float and a downstream consumer
    quantizes it. When that consumer's quantize is the op's sole user, fold it
    into the op's (previously null) output qparams so the op emits the quantized
    value directly and the redundant quantize is removed. The quantize may itself
    fan out to several consumers; they all read the same quantized value and are
    simply repointed at the fused op.

    Requires the fused op to have a single user (the quantize): an op feeding both
    the quantize and a separate float consumer would need to keep a float path,
    which this case does not handle.
    """
    if len(node.users) != 1:
        return False
    quant_node = next(iter(node.users.keys()))
    if quant_node.target != _QUANT_PER_TENSOR:
        return False

    new_scale = get_arg(quant_node, "scale", float)
    new_zp = get_arg(quant_node, "zero_point", int)
    new_dtype = get_arg(quant_node, "dtype", torch.dtype)
    new_quant_min = get_arg(quant_node, "quant_min", int)
    new_quant_max = get_arg(quant_node, "quant_max", int)

    _set_qparams(
        exported_program,
        node,
        "out",
        scale=new_scale,
        zero_point=new_zp,
        dtype=new_dtype,
        quant_min=new_quant_min,
        quant_max=new_quant_max,
    )

    quant_node.replace_all_uses_with(node)
    graph.erase_node(quant_node)

    # The fused op now carries the quantized value, so its dtype follows the
    # absorbed quantize (e.g. float32 → int8).
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        node.meta["val"] = val.to(new_dtype)

    return True


def _set_qparams(
    ep: ExportedProgram,
    node: fx.Node,
    prefix: str,
    *,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    quant_min: int,
    quant_max: int,
) -> None:
    """Replace a fused_quant op's qparams with new lifted constants.

    Lifted as ``CONSTANT_TENSOR`` to match how ``FusedQuantFusion`` lifts per-tensor
    qparams: its ``_scale``/``_zero_point`` module attributes are plain
    ``setattr`` tensors, which export classifies as lifted tensor constants (the
    graph's ``c__scale*`` placeholders confirm this -- ``c_`` is the
    CONSTANT_TENSOR prefix). Using a different kind here (e.g. BUFFER) would give
    the new node a ``b__scale`` name whose fqn ``_scale`` still collides with the
    fusion's ``ep.constants`` entry -- a cross-kind fqn clash node-name
    uniquification can't catch, tripping add_constant's assert.
    """
    new_scale_node = add_constant(
        ep,
        "_scale",
        torch.tensor(scale, dtype=torch.float32),
        node,
        InputKind.CONSTANT_TENSOR,
    )
    new_zp_node = add_constant(
        ep,
        "_zero_point",
        torch.tensor(zero_point, dtype=torch.int64),
        node,
        InputKind.CONSTANT_TENSOR,
    )

    set_arg(node, f"{prefix}_scale", new_scale_node)
    set_arg(node, f"{prefix}_zero_point", new_zp_node)
    set_arg(node, f"{prefix}_dtype", dtype)
    set_arg(node, f"{prefix}_quant_min", quant_min)
    set_arg(node, f"{prefix}_quant_max", quant_max)


def _try_absorb_input(
    exported_program: ExportedProgram,
    consumer: fx.Node,
    input_name: str,
    graph: fx.Graph,
) -> bool:
    """Absorb an upstream dequantize→quantize into fused input qparams."""
    consumer_qparams = get_qparams_from_node(consumer, input_name)
    if consumer_qparams is None or not consumer_qparams.is_per_tensor():
        return False

    consumer_input = get_arg(consumer, input_name, Optional[fx.Node])
    if consumer_input is None:
        return False
    quantize, passthrough_chain = _find_quant_through_input_passthrough(consumer_input)
    if quantize is None or len(quantize.users) != 1:
        return False

    dequantize = get_arg(quantize, "input", Optional[fx.Node])
    if dequantize is None or dequantize.target != _DEQUANT_PER_TENSOR:
        return False

    replacement_qparams = _compose_input_qparams(
        exported_program, dequantize, quantize, consumer, input_name
    )
    if replacement_qparams is None:
        return False
    replacement_scale, replacement_zp = replacement_qparams

    source = get_arg(dequantize, "input", Optional[fx.Node])
    if source is None:
        return False

    if passthrough_chain:
        passthrough_chain[-1].replace_input_with(quantize, source)
        source_dtype = source.meta["val"].dtype
        for passthrough in passthrough_chain:
            passthrough.meta["val"] = passthrough.meta["val"].to(source_dtype)
    else:
        set_arg(consumer, input_name, source)
    _set_qparams(
        exported_program,
        consumer,
        input_name,
        scale=replacement_scale,
        zero_point=replacement_zp,
        # Input dtype is the dequantized compute dtype; the source tensor carries
        # its integer storage dtype.
        dtype=dequantize.meta["val"].dtype,
        quant_min=get_arg(dequantize, "quant_min", int),
        quant_max=get_arg(dequantize, "quant_max", int),
    )
    if not quantize.users:
        graph.erase_node(quantize)
        if not dequantize.users:
            graph.erase_node(dequantize)
    return True


def _try_absorb_inputs(
    exported_program: ExportedProgram,
    consumer: fx.Node,
    graph: fx.Graph,
) -> bool:
    assert isinstance(consumer.target, EdgeOpOverload)
    input_names, _ = split_fused_arg_names(consumer.target)
    modified = False
    for input_name in input_names:
        modified |= _try_absorb_input(exported_program, consumer, input_name, graph)
    return modified


class QuantAbsorptionPass(ExportedProgramPassBase):
    """Absorb adjacent quantization boundaries into fused_quant operators.

    On the output side, when a fused_quant op with per-tensor output quantization
    feeds into a dequant→quant pair, this pass composes the producer, dequantize,
    and downstream quantize qparams into the fused op's output qparams. When the
    producer and dequantize qparams match, this reduces to the downstream target.

    On the input side, when a dequant→quant pair feeds any fused_quant input with
    per-tensor qparams, this pass composes the dequantize, quantize, and consumer
    input qparams, then bypasses the pair. This handles graph inputs, which have
    no fused producer into which the quantization can be absorbed.

    It also absorbs a *bare* downstream quantize into a fused op whose output is
    currently unquantized (null out qparams), turning that op into a directly
    quantizing one. This is what folds the quantize that a consumer places on a
    fused_quant.embedding's float gather into the embedding's output qparams.

    Args:
        absorb_with_fork: When True, also absorb dequant→quant pairs where
            the dequant has additional users (a "forked" dequant). The dequant
            is kept for the other users but its qparams are updated to match
            the new output qparams. This changes (not necessarily worsens) the
            numerics of the float path through the dequant, but should improve
            numerics on the int path since it removes a redundant dq-q pair.
    """

    def __init__(self, absorb_with_fork: bool = False) -> None:
        super().__init__()
        self._absorb_with_fork = absorb_with_fork

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph
        modified = False

        for node in list(graph.nodes):
            if not _is_fused_quant_op(node):
                continue
            if _has_per_tensor_out_qparams(node):
                modified |= _try_absorb_output(
                    exported_program, node, graph, self._absorb_with_fork
                )
            elif _has_unquantized_out(node):
                modified |= _try_absorb_quant_into_unquantized_out(
                    exported_program, node, graph
                )
            modified |= _try_absorb_inputs(exported_program, node, graph)

        if modified:
            constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)
