# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch

from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops

from later.unittest import TestCase


class TestIdmaOps(TestCase):
    def test_idma_load_store_wait(self) -> None:
        """Check that the idma load/store/wait ops are registered correctly."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(2, 7, dtype=torch.float32))
        load = builder.call_operator(
            op=exir_ops.edge.cadence.idma_load.default, args=(x,)
        )
        wait = builder.call_operator(
            op=exir_ops.edge.cadence.idma_wait.default, args=(load,)
        )
        store = builder.call_operator(
            op=exir_ops.edge.cadence.idma_store.default, args=(wait,)
        )
        wait2 = builder.call_operator(
            op=exir_ops.edge.cadence.idma_wait.default, args=(store,)
        )
        builder.output([wait2])
