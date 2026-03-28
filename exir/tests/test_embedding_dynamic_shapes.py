# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from torch import nn
from torch.export import Dim
from torch.export._patches import register_lstm_while_loop_decomposition

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime


class EmbeddingLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return x


class TestEmbeddingDynamicShapes(unittest.TestCase):

    def _export_model(self, export_seq_len: int):
        model = EmbeddingLSTMModel().eval()
        example_input = torch.randint(0, 100, (1, export_seq_len), dtype=torch.long)
        tokens = Dim("tokens", min=1, max=128)
        dynamic_shapes = ({1: tokens},)

        with register_lstm_while_loop_decomposition():
            exported = torch.export.export(
                model, (example_input,), dynamic_shapes=dynamic_shapes
            )

        edge = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        )
        return edge.to_executorch()

    def test_inference_different_seq_len(self):
        """Core regression: model exported at seq=16 must run at seq=5."""
        program = self._export_model(export_seq_len=16)

        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            f_name = f.name

        try:
            program.write_to_file(f_name)
            runtime = Runtime.get()
            loaded = runtime.load_program(f_name)
            method = loaded.load_method("forward")

            for seq_len in [1, 5, 16, 32, 128]:
                inp = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
                out = method.execute((inp,))
                self.assertEqual(
                    out[0].shape,
                    (1, seq_len, 64),
                    f"Wrong output shape at seq_len={seq_len}",
                )
        finally:
            os.unlink(f_name)

    def test_embedding_output_shape_is_symbolic(self):
        """Embedding output dim 1 must be SymInt after LSTM decomp."""
        model = EmbeddingLSTMModel().eval()
        example_input = torch.randint(0, 100, (1, 16), dtype=torch.long)
        tokens = Dim("tokens", min=1, max=128)

        with register_lstm_while_loop_decomposition():
            exported = torch.export.export(
                model, (example_input,), dynamic_shapes=({1: tokens},)
            )

        # Apply our pass directly and check metadata
        from executorch.exir._passes.fix_embedding_symbolic_shapes import (
            FixEmbeddingSymbolicShapes,
        )
        fixed = exported._transform(FixEmbeddingSymbolicShapes())

        for node in fixed.graph.nodes:
            if node.op == "call_function" and "embedding" in str(node.target):
                fake_val = node.meta.get("val")
                self.assertIsNotNone(fake_val)
                self.assertIsInstance(
                    fake_val.shape[1],
                    torch.SymInt,
                    "Embedding output dim 1 must be SymInt after fix pass",
                )


if __name__ == "__main__":
    unittest.main()
