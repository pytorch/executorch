# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guard for the LSTMCell-gate lowering unblocker.

`nn.LSTMCell` is exported as a single high-level op, so the quantizer never
sees the gate sigmoids/tanhs and they end up unannotated in the edge graph.
The lowering pass correctly skips them, but the user-facing effect is that
Silero VAD's LSTM gates stay in aten even after the quantized_activation
op lands.

Unblocking this requires a pre-annotation decompose pass that splits the
LSTMCell into linear + split + sigmoid + tanh + add + mul *before* the
quantizer annotates. When that lands, the four sigmoids + one tanh inside
this test's LSTMCell will lower to cortex_m.quantized_activation and this
test will pass -- at which point the xfail can be removed and the Silero
expectations updated.
"""

import pytest
import torch

from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.mark.xfail(
    reason="nn.LSTMCell is captured as a high-level op at export, so the "
    "quantizer doesn't annotate the gate activations. Needs a "
    "pre-annotation decompose pass to unblock.",
    strict=True,
)
def test_lstm_cell_gates_lower():
    hidden = 8
    model = torch.nn.LSTMCell(hidden, hidden).eval()
    x = torch.randn(1, hidden)
    h = torch.zeros(1, hidden)
    c = torch.zeros(1, hidden)

    tester = CortexMTester(model, (x, (h, c)))
    tester.quantize(None).export().to_edge().run_passes()

    gm = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    quantized_activations = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.cortex_m.quantized_activation.default
    ]
    # An LSTMCell has 3 sigmoid gates (i, f, o) + 1 tanh gate (g) + 1 output
    # tanh = 5 activation calls; all should lower once the decompose pass
    # makes them visible to the quantizer.
    assert len(quantized_activations) == 5, (
        f"expected 5 quantized_activation nodes (3 sigmoid gates + 2 tanh), "
        f"got {len(quantized_activations)}"
    )
