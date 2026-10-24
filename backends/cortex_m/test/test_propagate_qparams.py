# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.cortex_m.passes.propagate_qparams_pass import (
    PropagateQParamsPass,
)
from executorch.backends.cortex_m.utils import lowered_output_dtype
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, GraphModule, Node


_INT8 = QuantArgs(scale=0.5, zp=-3, qmin=-128, qmax=127, dtype=torch.int8)
_INT16 = QuantArgs(scale=0.25, zp=0, qmin=-32768, qmax=32767, dtype=torch.int16)

_VIEW_COPY = exir_ops.edge.aten.view_copy.default
_PERMUTE_COPY = exir_ops.edge.aten.permute_copy.default
_LAYOUT_PERMUTE_COPY = exir_ops.edge.channels_last.permute_copy.default
_UNSQUEEZE_COPY = exir_ops.edge.aten.unsqueeze_copy.default
_ADD = exir_ops.edge.aten.add.Tensor


def _fake(shape, dtype=torch.float32) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype)


class PropagateQParamsPassTest(unittest.TestCase):
    """The pass gives rearranging ops the qparams of the tensor they move.

    Each graph is built by hand rather than exported: the condition under test
    is a combination of node target and meta that only arises mid-pipeline,
    after FoldAndAnnotateQParamsPass has annotated the computing ops and left
    the plumbing with the pre-quantization fake tensor.
    """

    def setUp(self) -> None:
        self.graph = Graph()
        self.x: Node = self.graph.placeholder("x")
        self.x.meta["val"] = _fake((1, 4, 8), torch.int8)

    def _call(self, target, *args, val=None, qparams=None) -> Node:
        node = self.graph.call_function(target, args)
        node.meta["val"] = _fake((1, 4, 8)) if val is None else val
        if qparams is not None:
            node.meta["output_qparams"] = {0: qparams}
        return node

    def _run(self) -> bool:
        self.graph.output(())
        module = GraphModule(torch.nn.Module(), self.graph)
        return PropagateQParamsPass().call(module).modified

    def test_a_view_inherits_its_producers_qparams(self) -> None:
        producer = self._call(_ADD, self.x, self.x, qparams=_INT8)
        view = self._call(_VIEW_COPY, producer, [1, 32])

        self.assertTrue(self._run())
        self.assertEqual(view.meta["output_qparams"][0], _INT8)
        # The fake tensor is the stale fp32 one the trace left; the point of the
        # pass is that the dtype no longer has to be read off it.
        self.assertEqual(view.meta["val"].dtype, torch.float32)
        self.assertEqual(lowered_output_dtype(view), torch.int8)

    def test_a_chain_of_plumbing_resolves_in_one_pass(self) -> None:
        """The CWW shape: conv -> layout copy -> view -> permute -> view.

        Topological order is what makes one sweep enough -- each node reads a
        producer the same sweep has already annotated.
        """
        producer = self._call(_ADD, self.x, self.x, qparams=_INT8)
        layout = self._call(_LAYOUT_PERMUTE_COPY, producer, [0, 2, 1])
        view = self._call(_VIEW_COPY, layout, [1, 32])
        permute = self._call(_PERMUTE_COPY, view, [0, 2, 1])
        tail = self._call(_UNSQUEEZE_COPY, permute, 0)

        self.assertTrue(self._run())
        for node in (layout, view, permute, tail):
            self.assertEqual(
                lowered_output_dtype(node), torch.int8, f"{node.name} not resolved"
            )

    def test_existing_qparams_are_not_overwritten(self) -> None:
        producer = self._call(_ADD, self.x, self.x, qparams=_INT8)
        view = self._call(_VIEW_COPY, producer, [1, 32], qparams=_INT16)

        self.assertFalse(self._run())
        self.assertEqual(view.meta["output_qparams"][0], _INT16)

    def test_an_unquantized_producer_propagates_nothing(self) -> None:
        """A view in an fp32 stretch stays fp32 -- there is nothing to inherit."""
        producer = self._call(_ADD, self.x, self.x)
        view = self._call(_VIEW_COPY, producer, [1, 32])

        self.assertFalse(self._run())
        self.assertNotIn("output_qparams", view.meta)
        self.assertEqual(lowered_output_dtype(view), torch.float32)

    def test_a_computing_op_is_not_transparent(self) -> None:
        """Only rearranging is dtype-preserving; an add is free to requantize."""
        producer = self._call(_ADD, self.x, self.x, qparams=_INT8)
        consumer = self._call(_ADD, producer, producer)

        self.assertFalse(self._run())
        self.assertNotIn("output_qparams", consumer.meta)
