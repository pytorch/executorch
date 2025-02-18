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
from executorch.exir.backend.utils import format_delegated_graph


class TestCoreMLPartitioner(unittest.TestCase):
    edge_compile_config = executorch.exir.EdgeCompileConfig()

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
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)

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
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)
        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(
            CoreMLPartitioner(
                skip_ops_for_coreml_delegation=["aten.convolution.default"]
            )
        )

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "aten.convolution.default",
            "executorch_call_delegate",
            "getitem",
        ]

    def test_ops_to_not_decompose(self):
        class Model(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, attn_mask=mask
                )

        model = Model()
        model.eval()

        batch_size = 1
        n_heads = 12
        seq_len = 1
        max_seq_length = 32
        embedding_dim = 16
        q = torch.randn(batch_size, n_heads, seq_len, embedding_dim)
        k = torch.randn(batch_size, n_heads, max_seq_length, embedding_dim)
        v = torch.randn(batch_size, n_heads, max_seq_length, embedding_dim)
        mask = torch.randn(seq_len, max_seq_length)
        example_inputs = (q, k, v, mask)
        ep = torch.export.export(model, example_inputs)
        coreml_partitioner = CoreMLPartitioner()

        # Using to_edge_transform_and_lower, we expect SDPA will be preserved and show up in delegated graph
        edge_program_manager = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[coreml_partitioner]
        )
        self.assertTrue(
            "executorch.exir.dialects.edge._ops.aten.scaled_dot_product_attention.default"
            in format_delegated_graph(
                edge_program_manager.exported_program().graph_module
            )
        )

        # Using to_edge flow, we expect SDPA will be decomposed and not show up in delegated graph
        edge_program_manager2 = executorch.exir.to_edge(ep)
        edge_program_manager2.to_backend(coreml_partitioner)
        self.assertTrue(
            "executorch.exir.dialects.edge._ops.aten.scaled_dot_product_attention.default"
            not in format_delegated_graph(
                edge_program_manager2.exported_program().graph_module
            )
        )

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
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)

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
    test_runner.test_ops_to_not_decompose()
    test_runner.test_buffer()
