# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

from typing import List

import torch
from executorch.exir import EdgeCompileConfig, to_edge

from .replace_custom_ops_with_aten_ops_pass import ReplaceCustomOpsWithAtenOpsPass


class TestPasses(unittest.TestCase):
    def test_replace_custom_ops_with_aten_ops_pass(self) -> None:
        from executorch.extension.llm.custom_ops import preprocess_custom_ops  # noqa

        class Pad(torch.nn.Module):
            def forward(self, x: torch.Tensor, padding: List[int]) -> torch.Tensor:
                return torch.ops.preprocess.pad.default(x, padding)

        pad = Pad()

        image_tensor = torch.ones([3, 4, 5])
        padding = [0, 2, 0, 1]

        edge_prog = to_edge(
            torch.export.export(pad, (image_tensor, padding), strict=False),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        # Check that the custom op exists in the graph, and aten op does not.
        edge_nodes = [node.name for node in edge_prog.exported_program().graph.nodes]
        assert "constant_pad_nd" not in edge_nodes
        assert "preprocess_pad_default" in edge_nodes

        edge_prog = edge_prog.transform([ReplaceCustomOpsWithAtenOpsPass()])

        # After running replace_custom_ops_with_aten_ops pass, the custom op
        # should be replaced with aten op.
        post_transform_nodes = [
            node.name for node in edge_prog.exported_program().graph.nodes
        ]
        assert "constant_pad_nd" in post_transform_nodes
        assert "preprocess_pad_default" not in post_transform_nodes
