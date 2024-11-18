# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import coremltools as ct

import executorch.exir

import torch
import torchvision

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner


class TestCoreMLPartitioner(unittest.TestCase):

    # TODO(T182928844): Delegate dim order op to backend.
    edge_compile_config = executorch.exir.EdgeCompileConfig(_skip_dim_order=True)

    def test_add_sub_skip_mm(self):
        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        model = Model()
        model.eval()

        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        exir_program_aten = torch.export.export(model, example_inputs)

        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(
            CoreMLPartitioner(skip_ops_for_coreml_delegation=["aten.mm.default"])
        )

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "aten.mm.default",
            "executorch_call_delegate",
            "getitem",
            "aten.mm.default",
            "executorch_call_delegate",
            "getitem",
        ]

    def test_vit_skip_conv(self):
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        model.eval()

        example_inputs = (torch.randn(1, 3, 224, 224),)
        exir_program_aten = torch.export.export(model, example_inputs)
        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(
            CoreMLPartitioner(
                skip_ops_for_coreml_delegation=["aten.convolution.default"]
            )
        )

        conv_block = ["aten.convolution.default", "executorch_call_delegate"]
        safe_softmax_block = [
            "getitem",
            "getitem",
            "getitem",
            "getitem",
            "aten.any.dim",
            "executorch_call_delegate",
        ]
        final_block = ["getitem"]
        total = conv_block + 12 * safe_softmax_block + final_block

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == total

    def test_buffer(self):
        embedding_dim = 3
        max_seq_len = 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "cache",
                    torch.zeros((max_seq_len, embedding_dim), dtype=torch.float32),
                )

            def forward(self, q, k_val, input_pos):
                q_T = q.transpose(0, 1)
                k = torch.ops.aten.index_put_(self.cache, [input_pos, None], k_val)
                attn = k.mm(q_T)
                return attn

        model = Model()
        model.eval()

        q = torch.randn((1, embedding_dim))
        k_val = torch.randn((1, embedding_dim))
        input_pos = torch.tensor([0])
        example_inputs = (q, k_val, input_pos)
        exir_program_aten = torch.export.export(model, example_inputs)

        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        partitioner = CoreMLPartitioner(compile_specs=compile_specs)
        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(partitioner)

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "executorch_call_delegate",
            "getitem",
        ]


if __name__ == "__main__":
    test_runner = TestCoreMLPartitioner()
    test_runner.test_add_sub_skip_mm()
    test_runner.test_vit_skip_conv()
    test_runner.test_buffer()
