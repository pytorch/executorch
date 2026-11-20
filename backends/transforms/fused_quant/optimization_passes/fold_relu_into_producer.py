# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.dialects.fused_quant.ops import QuantParamsStruct
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch.export import ExportedProgram

# Producers that quantize a freshly computed result, so they can quantize it to
# any qparams at no extra cost.
_PRODUCERS: frozenset[EdgeOpOverload] = frozenset(
    {
        exir_ops.edge.fused_quant.conv1d.default,
        exir_ops.edge.fused_quant.conv2d.default,
        exir_ops.edge.fused_quant.conv3d.default,
        exir_ops.edge.fused_quant.convolution.default,
        exir_ops.edge.fused_quant.convolution_channels_last.default,
        exir_ops.edge.fused_quant.linear.default,
    }
)

_PASSTHROUGH: frozenset[EdgeOpOverload] = frozenset(
    {
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    }
)


def _value(
    exported_program: ExportedProgram, qparams: QuantParamsStruct[fx.Node]
) -> tuple[float, int, int, int]:
    return (
        get_scale(exported_program, qparams),
        get_zero_point(exported_program, qparams),
        qparams.quant_min,
        qparams.quant_max,
    )


def _producer(relu: fx.Node) -> fx.Node | None:
    """The weighted op whose only reader, through data movement, is ``relu``."""
    cursor = get_arg(relu, "inp", fx.Node)
    while cursor.target in _PASSTHROUGH and len(cursor.users) == 1:
        cursor = cursor.args[0]
    if (
        cursor.op != "call_function"
        or cursor.target not in _PRODUCERS
        or len(cursor.users) != 1
        or not isinstance(cursor.meta.get("val"), torch.Tensor)
    ):
        return None
    return cursor


def _fold(exported_program: ExportedProgram, relu: fx.Node) -> bool:
    producer = _producer(relu)
    if producer is None:
        return False
    produced = get_qparams_from_node(producer, "out")
    read = get_qparams_from_node(relu, "inp")
    written = get_qparams_from_node(relu, "out")
    if (
        produced is None
        or read is None
        or written is None
        or not all(q.is_per_tensor() for q in (produced, read, written))
        or produced.dtype != written.dtype
        # The relu must read the producer's bytes as they were written.
        or _value(exported_program, produced) != _value(exported_program, read)
    ):
        return False

    written_value = _value(exported_program, written)
    clamps_nothing = written_value[1] == written.quant_min
    if written_value == _value(exported_program, produced) and not clamps_nothing:
        return False

    set_arg(producer, "out_scale", written.scale)
    set_arg(producer, "out_zero_point", written.zero_point)
    set_arg(producer, "out_quant_min", written.quant_min)
    set_arg(producer, "out_quant_max", written.quant_max)
    if clamps_nothing:
        relu.replace_all_uses_with(get_arg(relu, "inp", fx.Node))
        relu.graph.erase_node(relu)
    else:
        set_arg(relu, "inp_scale", written.scale)
        set_arg(relu, "inp_zero_point", written.zero_point)
        set_arg(relu, "inp_quant_min", written.quant_min)
        set_arg(relu, "inp_quant_max", written.quant_max)
    return True


class FoldReluIntoProducer(ExportedProgramPassBase):
    """Let the conv / linear that feeds a relu quantize straight to its output.

    A relu that reads its producer at ``A`` and writes ``B`` is a clamp and a
    rescale. The producer quantizes a freshly computed result anyway, so it can
    target ``B`` directly: it takes the relu's output qparams and the relu then
    reads and writes ``B``. Where ``B``'s zero point is its ``quant_min``,
    quantizing to ``B`` already saturates every negative value to zero, so the
    relu does nothing and is removed. Otherwise it stays, as a clamp that needs
    no rescale.

    Upstream ``QuantAbsorptionPass`` produces the ``A != B`` relu by composing
    the requantize after an activation into the activation's output, since it
    cannot reach past the activation into the producer. This finishes that job.

    The result drops the rounding to ``A``: the relu's output is ``Q_B(x)``
    instead of ``Q_B(relu(DQ_A(Q_A(x))))``, so it can differ from the graph
    before the pass by the error that intermediate step introduced.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph
        modified = False
        for relu in list(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.relu.default
            )
        ):
            modified |= _fold(exported_program, relu)
        if modified:
            graph.eliminate_dead_code()
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
