import unittest

import torch
from executorch.backends.qualcomm._passes import (
    ConvertBmmToMatmul,
    ConvertMhaToSha,
    InsertReshapeForReduceOps,
    RemoveRedundancy,
)
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.tests.models import TopKandIndex
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import to_edge
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
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
        args.max_context_len = 128
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

    def test_resolve_debug_handle(self):
        name_handle_map = {
            "aten_topk_default": 1,
            "getitem": 1,
            "getitem_1": 1,
            "aten_view_copy_default": 2,
            "aten_index_tensor": 3,
            "aten_add_tensor": 4,
        }
        module = TopKandIndex()  # noqa: F405
        sample_input = (torch.randn(3, 10),)

        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,  # Random soc_model
            backend_options=backend_options,
            dump_intermediate_outputs=True,
        )

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module,
            sample_input,
            compiler_spec,
            generate_etrecord=True,
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch()
        etrecord = exec_prog_mgr.get_etrecord()
        debug_handle_size = len(etrecord._debug_handle_map["forward"][0])
        self.assertEqual(
            len(name_handle_map),
            debug_handle_size,
            f"Number of handles does not match, expecting: {len(name_handle_map)}, but get: {debug_handle_size}",
        )
        after_edge_pass_ep = etrecord.graph_map["edge_after_transform/forward"]

        for node in after_edge_pass_ep.graph.nodes:
            if node.name in name_handle_map:
                expected_handle = name_handle_map.pop(node.name)
                node_handle = node.meta[DEBUG_HANDLE_KEY]
                self.assertEqual(
                    expected_handle,
                    node_handle,
                    f"{node.name} is expecting a handle id {expected_handle}, but got {node_handle}.",
                )
        self.assertEqual(
            len(name_handle_map),
            0,
            f"Following nodes did not find a match in the graph: {name_handle_map.keys()}",
        )


if __name__ == "__main__":
    unittest.main()
