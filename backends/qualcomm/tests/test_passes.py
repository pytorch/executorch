import unittest

import torch
from executorch.backends.qualcomm._passes import (
    ConvertBmmToMatmul,
    ConvertMhaToSha,
    InsertReshapeForReduceOps,
    RemoveRedundancy,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TestPasses(unittest.TestCase):
    def test_insert_reshape_for_argmax(self):
        class ArgmaxModule(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=None)

        mod = ArgmaxModule()

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        ep = torch.export.export(mod, (x,))
        # Run original module for reference
        ref = mod(x)

        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 0, "Reshape node not inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        InsertReshapeForReduceOps()(ep.graph_module)

        out = ep.graph_module(x)

        # Check graph structure: argmax should take a reshape as input
        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 1, "Reshape node should be inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        argmax_node = argmax_nodes[0]
        self.assertEqual(argmax_node.args[1], 0, "Argmax dim not set to 0")

        # Execute new graph and compare with reference
        out = ep.graph_module(x)
        self.assertTrue(
            torch.equal(*out, ref), f"Output mismatch: got {out}, expected {ref}"
        )

    def test_mha_to_sha(self):
        from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
        from executorch.examples.models.llama.model_args import ModelArgs
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            CausalAttentionMask,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaAttention,
        )

        # Initailize model config
        args = ModelArgs()
        args.max_seq_len = 128
        args.ar_len = 32
        args.use_kv_cache = True
        args.dim = 32
        args.n_heads = 8
        args.n_kv_heads = 8
        args.n_layers = 2
        args.head_dim = args.dim // args.n_heads
        mod = convert_linear_to_conv2d(LlamaAttention(0, args, True))

        # Prepare inputs
        hidden_states = torch.randn(args.max_batch_size, args.ar_len, args.dim)
        freqs_cos = torch.randn(args.ar_len, 1)
        freqs_sin = torch.randn(args.ar_len, 1)
        atten_mask = CausalAttentionMask(
            args.max_batch_size, args.ar_len, args.max_seq_len
        )
        k_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.head_dim,
            args.max_seq_len - args.ar_len,
        )

        v_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.max_seq_len - args.ar_len,
            args.head_dim,
        )
        sample_input = (
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask.mask,
            k_cache,
            v_cache,
        )

        # Run original module for reference
        refs = mod(*sample_input)

        # Export the module and convert linear to conv2d
        edge_program = to_edge(torch.export.export(mod, sample_input))
        new_ep = edge_program.exported_program()

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # WQ, WK, WV, O
        self.assertTrue(len(conv_nodes) == 4, "Convolution nodes missing")

        # Convert MHA to SHA
        # This is a simplified version of what happens in the full pipeline to test the core functionality
        graph_module = RemoveRedundancy(quantization_capture=False)(
            new_ep.graph_module
        ).graph_module
        graph_module = ConvertBmmToMatmul()(graph_module).graph_module
        graph_module = ConvertMhaToSha(new_ep)(graph_module).graph_module

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # Check graph structure: WQ, WK, WV should be converted to SHA
        self.assertTrue(len(conv_nodes) == 25, "Convolution nodes should be splited")

        # Execute new graph and compare with reference
        outs = graph_module(
            *new_ep.state_dict.values(), *new_ep.constants.values(), *sample_input
        )
        for i, (out, ref) in enumerate(zip(outs, refs)):
            self.assertTrue(
                torch.allclose(out, *ref, rtol=1e-6, atol=1e-6),
                f"Output {i} mismatch: got {out}, expected {ref}",
            )

    def test_arange_dtype_from_kwargs(self):
        """Test that op_arange builder respects dtype from node kwargs."""
        from collections import defaultdict

        import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
        from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
        from executorch.backends.qualcomm.builders.node_visitor_manager import (
            get_node_visitors,
        )
        from executorch.backends.qualcomm.utils.utils import capture_program

        class ArangeWithDtype(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.arange(0, 10, 1, dtype=self.dtype) + x

        # Map torch dtypes to expected QNN data types
        dtype_to_qnn = {
            torch.float32: PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            torch.int64: PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_64,
            torch.int32: PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
        }

        test_cases = [
            (torch.float32, dtype_to_qnn[torch.float32]),
            (torch.int64, dtype_to_qnn[torch.int64]),
            (torch.int32, dtype_to_qnn[torch.int32]),
            (None, dtype_to_qnn[torch.int64]),  # None uses torch default (int64)
        ]

        for input_dtype, expected_qnn_dtype in test_cases:
            with self.subTest(input_dtype=input_dtype):
                mod = ArangeWithDtype(input_dtype)
                # Use appropriate input tensor dtype
                if input_dtype == torch.float32:
                    x = torch.zeros(10, dtype=torch.float32)
                else:
                    x = torch.zeros(10, dtype=torch.int64)

                sample_input = (x,)
                delegated_program = capture_program(mod, sample_input)

                # Transform graph through QNN pass manager
                graph_module = QnnPassManager().transform_for_preprocess_pipeline(
                    delegated_program.exported_program
                )

                # Get node visitors (builders)
                nodes_to_wrappers = defaultdict(dict)
                node_visitors = get_node_visitors(
                    delegated_program.exported_program, enable_tensor_dump=False
                )

                # Find and process the arange node through the builder
                for node in graph_module.graph.nodes:
                    if node.op == "call_function" and "arange" in node.target.__name__:
                        # Invoke the Arange builder's define_node
                        node_visitors[node.target.__name__].define_node(
                            node, nodes_to_wrappers
                        )

                        # Check that a tensor wrapper was created
                        self.assertIn(
                            node.name,
                            nodes_to_wrappers,
                            f"No tensor wrapper created for arange node",
                        )

                        # Get the tensor wrapper and verify its dtype
                        tensor_wrapper = nodes_to_wrappers[node.name][0]
                        actual_dtype = tensor_wrapper.GetDataType()

                        self.assertEqual(
                            actual_dtype,
                            expected_qnn_dtype,
                            f"QNN dtype mismatch for input_dtype={input_dtype}: "
                            f"got {actual_dtype}, expected {expected_qnn_dtype}",
                        )
                        break
                else:
                    self.fail("No arange node found in graph")


if __name__ == "__main__":
    unittest.main()
