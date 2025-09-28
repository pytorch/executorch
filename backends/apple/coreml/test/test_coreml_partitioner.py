# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import copy
import sys
import unittest

import coremltools as ct

import executorch.exir

import torch
import torchvision

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner, SingleOpCoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.backend.utils import format_delegated_graph
from torchao.quantization import IntxWeightOnlyConfig, PerAxis, PerGroup, quantize_

if IS_VALID_TEST_RUNTIME:
    from executorch.runtime import Runtime


@torch.library.custom_op("unsupported::linear", mutates_args=())
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.linear.default(x, w, b)


@torch.library.register_fake("unsupported::linear")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.linear.default(x, w, b)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


_TEST_RUNTIME = (sys.platform == "darwin") and not is_fbcode()
if _TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestCoreMLPartitioner(unittest.TestCase):
    edge_compile_config = executorch.exir.EdgeCompileConfig()

    def _coreml_partitioner(self, *, minimum_deployment_target=ct.target.iOS18):
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=minimum_deployment_target
        )
        return CoreMLPartitioner(compile_specs=compile_specs)

    def _single_op_coreml_partitioner(self, *, minimum_deployment_target=ct.target.iOS18):
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=minimum_deployment_target
        )
        return SingleOpCoreMLPartitioner(compile_specs=compile_specs)

    def _compare_outputs(self, executorch_program, eager_program, example_inputs):
        if not IS_VALID_TEST_RUNTIME:
            return
        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        et_outputs = method.execute(*example_inputs)[0]
        eager_outputs = eager_program(*example_inputs)
        self.assertTrue(
            torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02)
        )

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
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v, mask):
                out = torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, attn_mask=mask
                )

                # triu/tril should be ignored by do_not_decompose
                # because otherwise they fail during CoreML lowering
                offset1 = torch.triu(mask, diagonal=1)
                offset2 = torch.tril(mask)
                offset = offset1 + offset2
                offset = torch.sum(offset)

                # Add non-functional and alias ops
                # These will be removed by ExecuTorch in non-decomposition
                # table because they cannot be functionalized
                out = out.transpose(1, 2)
                out = out.view(1, -1)
                out = out.permute(0, 1)
                out = out.add_(1.0)
                out = out.mul_(2.0)
                out = out.div_(3.0)
                out = out.sub_(4.0)
                out = torch.ops.aten.view_copy.default(out, (-1,))
                out = out.select(0, 0)
                return out + offset

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
        ep = torch.export.export(model, example_inputs, strict=True)
        self.assertTrue(
            "torch.ops.aten.triu.default" in ep.graph_module.code,
        )
        self.assertTrue(
            "torch.ops.aten.tril.default" in ep.graph_module.code,
        )

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

    def test_lower_full_graph(self):
        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                out = torch.ops.aten.linear.default(a, x, b)
                out2 = torch.ops.unsupported.linear.default(out, x, b)
                return out2

        model = Model()
        model.eval()

        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)
        edge_program_manager = executorch.exir.to_edge(exir_program_aten)
        edge_program_manager2 = copy.deepcopy(edge_program_manager)

        delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner())

        for node in delegated_program_manager.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "unsupported.linear.default",
                    "executorch_call_delegate",
                    "getitem",
                ], node.target.__name__

        with self.assertRaises(NotImplementedError):
            edge_program_manager2.to_backend(CoreMLPartitioner(lower_full_graph=True))

    # TODO: enable this after bugs are fixed in ExecuTorch's partitioner
    # def test_symint_arg(self):
    #     class Model(torch.nn.Module):
    #         def forward(self, x, w, b, y):
    #             val = y.item()
    #             torch._check(val >= 0)
    #             torch._check(val < 2)
    #             out = torch.ops.aten.linear.default(x, w, b)
    #             out2 = out.relu()[val]
    #             return out2

    #     model = Model()
    #     model.eval()
    #     example_inputs = (
    #         torch.randn(2, 2),
    #         torch.randn(2, 2),
    #         torch.randn(2, 2),
    #         torch.tensor(2),
    #     )
    #     exir_program_aten = torch.export.export(model, example_inputs)

    #     edge_program_manager = executorch.exir.to_edge(exir_program_aten)

    #     delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner(skip_ops_for_coreml_delegation=["aten.scalar_tensor.default"]))

    #     # This op has symbolic args
    #     assert (
    #         "torch.ops.aten._assert_scalar.default"
    #         in delegated_program_manager.exported_program().graph_module.code
    #     )

    #     if _TEST_RUNTIME:
    #         et_prog = delegated_program_manager.to_executorch()
    #         runtime = Runtime.get()
    #         program = runtime.load_program(et_prog.buffer)
    #         method = program.load_method("forward")
    #         et_outputs = method.execute(*example_inputs)[0]
    #         eager_outputs = model(*example_inputs)
    #         self.assertTrue(torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02))

    def test_take_over_constant_data_false(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(50, 100)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()
        example_inputs = (torch.randn(2, 50),)
        exir_program_aten = torch.export.export(model, example_inputs)

        edge_program_manager = executorch.exir.to_edge_transform_and_lower(
            exir_program_aten,
            partitioner=[CoreMLPartitioner(take_over_constant_data=False)],
        )
        for node in edge_program_manager.exported_program().graph.nodes:
            if (
                node.op == "call_function"
                and node.target.__name__ == "executorch_call_delegate"
            ):
                break

        # lowered_module_0, x, p_linear_weight, p_linear_bias
        assert len(node.args) == 4

        if _TEST_RUNTIME:
            et_prog = edge_program_manager.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_prog.buffer)
            method = program.load_method("forward")
            et_outputs = method.execute(*example_inputs)[0]
            eager_outputs = model(*example_inputs)
            self.assertTrue(
                torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02)
            )

    def test_single_op_partitioner_basic(self):
        """Test that SingleOpCoreMLPartitioner creates individual delegates for each operation."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        model = SimpleModel()
        model.eval()
        example_inputs = (torch.randn(1, 10),)

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Test original partitioner - should create fewer delegates
        delegated_program_manager = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._coreml_partitioner()],
        )

        original_delegates = [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        # Test single op partitioner - should create more delegates
        delegated_program_manager_single = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        single_op_delegates = [
            node.target.__name__
            for node in delegated_program_manager_single.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        # SingleOpCoreMLPartitioner should create more individual delegates
        self.assertGreater(len(single_op_delegates), len(original_delegates))
        self.assertGreaterEqual(len(single_op_delegates), 3)  # At least linear1, relu, linear2

        # Test to_executorch and compare outputs
        et_prog_original = delegated_program_manager.to_executorch()
        et_prog_single_op = delegated_program_manager_single.to_executorch()

        self._compare_outputs(et_prog_original, model, example_inputs)
        self._compare_outputs(et_prog_single_op, model, example_inputs)


    def test_single_op_partitioner_skip_get_attr(self):
        """Test that get_attr nodes are properly skipped."""
        class ModelWithConstants(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("constant", torch.ones(5))
                self.linear = torch.nn.Linear(5, 3)

            def forward(self, x):
                x = x + self.constant
                return self.linear(x)

        model = ModelWithConstants()
        model.eval()
        example_inputs = (torch.randn(1, 5),)

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Should not fail due to get_attr nodes
        delegated_program_manager = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        # Check that we have delegates created
        delegates = [
            node for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        self.assertGreater(len(delegates), 0)

        # Test to_executorch
        et_prog = delegated_program_manager.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_single_op_partitioner_with_skipped_ops(self):
        """Test SingleOpCoreMLPartitioner with skip_ops_for_coreml_delegation."""
        class ModelWithMM(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                return z

        model = ModelWithMM()
        model.eval()
        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Create partitioner with skipped ops
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        partitioner = SingleOpCoreMLPartitioner(
            skip_ops_for_coreml_delegation=["aten.mm.default"],
            compile_specs=compile_specs
        )

        delegated_program_manager = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
        )

        call_functions = [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ]

        # Should have mm.default not delegated, but add.Tensor delegated
        self.assertIn("aten.mm.default", call_functions)
        self.assertIn("executorch_call_delegate", call_functions)

        # Test to_executorch
        et_prog = delegated_program_manager.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_single_op_partitioner_multiple_ops_separate_delegates(self):
        """Test that multiple different operations create separate call_delegate nodes."""
        class MultiOpModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 15)
                self.linear2 = torch.nn.Linear(15, 20)
                self.linear3 = torch.nn.Linear(20, 5)

            def forward(self, x):
                # This should create separate delegates for:
                # linear1, relu, linear2, sigmoid, linear3, tanh
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                x = torch.sigmoid(x)
                x = self.linear3(x)
                x = torch.tanh(x)
                return x

        model = MultiOpModel()
        model.eval()
        example_inputs = (torch.randn(1, 10),)

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Test SingleOpCoreMLPartitioner
        delegated_program_manager = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        delegates = [
            node for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        # Should have at least 6 separate delegates (3 linear + 3 activation functions)
        self.assertGreaterEqual(len(delegates), 6)

        # Compare with original partitioner which should create fewer delegates
        delegated_program_manager_orig = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._coreml_partitioner()],
        )

        original_delegates = [
            node for node in delegated_program_manager_orig.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        # SingleOp should create more delegates than original
        self.assertGreater(len(delegates), len(original_delegates))

        # Test to_executorch and compare outputs
        et_prog_single = delegated_program_manager.to_executorch()
        et_prog_orig = delegated_program_manager_orig.to_executorch()

        self._compare_outputs(et_prog_single, model, example_inputs)
        self._compare_outputs(et_prog_orig, model, example_inputs)

    def test_single_op_partitioner_conv_ops_separate_delegates(self):
        """Test that convolution operations create separate delegates."""
        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                x = self.conv2(x)
                x = torch.relu(x)
                return x

        model = ConvModel()
        model.eval()
        example_inputs = (torch.randn(1, 3, 32, 32),)

        exported_program = torch.export.export(model, example_inputs, strict=True)

        delegated_program_manager = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        delegates = [
            node for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function" and node.target.__name__ == "executorch_call_delegate"
        ]

        # Should have separate delegates for conv1, relu, max_pool2d, conv2, relu
        self.assertGreaterEqual(len(delegates), 5)

        # Test to_executorch
        et_prog = delegated_program_manager.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_single_op_partitioner_4bit_weight_only_linear(self):
        """Test 4-bit weight-only quantized linear layers with SingleOpCoreMLPartitioner."""
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(64, 128)  # Use sizes divisible by block size
                self.linear2 = torch.nn.Linear(128, 32)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        model = TestModel()
        model.eval()
        example_inputs = (torch.randn(1, 64),)

        # Apply 4-bit weight-only quantization to linear layers
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Test with SingleOpCoreMLPartitioner
        delegated_program_single = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        # Test with original CoreMLPartitioner
        delegated_program_orig = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._coreml_partitioner()],
        )

        # Check that both work
        for node in delegated_program_single.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        for node in delegated_program_orig.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        # Test to_executorch and compare outputs
        et_prog_single = delegated_program_single.to_executorch()
        et_prog_orig = delegated_program_orig.to_executorch()

        self._compare_outputs(et_prog_single, model, example_inputs)
        self._compare_outputs(et_prog_orig, model, example_inputs)

    def test_single_op_partitioner_4bit_embedding(self):
        """Test 4-bit weight-only quantized embedding layers with SingleOpCoreMLPartitioner."""
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 128)
                self.linear = torch.nn.Linear(128, 256)

            def forward(self, x):
                x = self.embedding(x)
                x = self.linear(x)
                return x

        model = TestModel()
        model.eval()
        example_inputs = (torch.LongTensor([0]),)

        # Apply 4-bit weight-only quantization to embedding layer
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Test with SingleOpCoreMLPartitioner
        delegated_program_single = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        # Test with original CoreMLPartitioner
        delegated_program_orig = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._coreml_partitioner()],
        )

        # Check that both work
        for node in delegated_program_single.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        for node in delegated_program_orig.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        # Test to_executorch and compare outputs
        et_prog_single = delegated_program_single.to_executorch()
        et_prog_orig = delegated_program_orig.to_executorch()

        self._compare_outputs(et_prog_single, model, example_inputs)
        self._compare_outputs(et_prog_orig, model, example_inputs)

    def test_single_op_partitioner_mixed_4bit_embedding_and_linear(self):
        """Test model with both 4-bit embedding and 4-bit linear quantization."""
        class MixedQuantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 128)
                self.linear1 = torch.nn.Linear(128, 256)
                self.linear2 = torch.nn.Linear(256, 128)

            def forward(self, x):
                x = self.embedding(x)
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        model = MixedQuantModel()
        model.eval()
        example_inputs = (torch.LongTensor([0]),)

        # Apply 4-bit quantization to embedding
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

        # Apply 4-bit quantization to linear layers
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            lambda m, fqn: isinstance(m, torch.nn.Linear),
        )

        exported_program = torch.export.export(model, example_inputs, strict=True)

        # Test with SingleOpCoreMLPartitioner
        delegated_program_single = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._single_op_coreml_partitioner()],
        )

        # Test with original CoreMLPartitioner
        delegated_program_orig = executorch.exir.to_edge_transform_and_lower(
            exported_program,
            partitioner=[self._coreml_partitioner()],
        )

        # Check that both work
        for node in delegated_program_single.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        for node in delegated_program_orig.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        # Test to_executorch and compare outputs
        et_prog_single = delegated_program_single.to_executorch()
        et_prog_orig = delegated_program_orig.to_executorch()

        self._compare_outputs(et_prog_single, model, example_inputs)
        self._compare_outputs(et_prog_orig, model, example_inputs)


if __name__ == "__main__":
    test_runner = TestCoreMLPartitioner()
    # Original CoreMLPartitioner tests
    test_runner.test_add_sub_skip_mm()
    test_runner.test_vit_skip_conv()
    test_runner.test_ops_to_not_decompose()
    test_runner.test_buffer()
    test_runner.test_lower_full_graph()
    # test_runner.test_symint_arg()
    test_runner.test_take_over_constant_data_false()

    # SingleOpCoreMLPartitioner tests
    test_runner.test_single_op_partitioner_basic()
    test_runner.test_single_op_partitioner_skip_get_attr()
    test_runner.test_single_op_partitioner_with_skipped_ops()
    test_runner.test_single_op_partitioner_multiple_ops_separate_delegates()
    test_runner.test_single_op_partitioner_conv_ops_separate_delegates()
    test_runner.test_single_op_partitioner_4bit_weight_only_linear()
    test_runner.test_single_op_partitioner_4bit_embedding()
    test_runner.test_single_op_partitioner_mixed_4bit_embedding_and_linear()
