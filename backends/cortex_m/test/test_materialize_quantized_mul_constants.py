# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cortex_m.passes.materialize_quantized_mul_constants_pass import (
    MaterializeQuantizedMulConstantsPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph


def _mark_quantized(node):
    node.meta["input_qparams"] = {0: object()}
    node.meta["output_qparams"] = {0: object()}


def test_accepts_single_quantized_mul_path():
    graph = Graph()

    full = graph.call_function(
        exir_ops.edge.aten.full.default,
        args=((1,), 0.5),
    )
    clone = graph.call_function(
        exir_ops.edge.aten.clone.default,
        args=(full,),
    )
    other = graph.placeholder("other")
    mul = graph.call_function(
        exir_ops.edge.aten.mul.Tensor,
        args=(other, clone),
    )
    _mark_quantized(mul)

    assert MaterializeQuantizedMulConstantsPass._feeds_single_quantized_mul(full)


def test_rejects_shared_constant_with_unrelated_consumer():
    graph = Graph()

    full = graph.call_function(
        exir_ops.edge.aten.full.default,
        args=((1,), 0.5),
    )
    other = graph.placeholder("other")

    mul = graph.call_function(
        exir_ops.edge.aten.mul.Tensor,
        args=(other, full),
    )
    _mark_quantized(mul)

    graph.call_function(
        exir_ops.edge.aten.add.Tensor,
        args=(other, full),
    )

    assert not MaterializeQuantizedMulConstantsPass._feeds_single_quantized_mul(full)
