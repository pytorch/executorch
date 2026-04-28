import unittest

import torch
from executorch.backends.qualcomm._passes import (
    AnnotateQuantAttrs,
    ConvertBmmToMatmul,
    ConvertMhaToSha,
    FoldQDQ,
    InsertIOQDQ,
    InsertRequantize,
    InsertReshapeForReduceOps,
    RemoveRedundancy,
)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.quantizer.rules import Q_ANNOTATION_KEY
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
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec


class TestPasses(unittest.TestCase):
    def _build_quantized_graph(self, module=None, sample_input=None):
        """Build a quantized graph through AnnotateQuantAttrs + FoldQDQ."""

        if module is None:

            class AddModule(torch.nn.Module):
                def forward(self, x):
                    return x + 1

            module = AddModule()

        if sample_input is None:
            sample_input = (torch.randn(1, 4),)

        module = module.eval()
        exported = torch.export.export(module, sample_input, strict=True).module()
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*sample_input)
        qdq_module = convert_pt2e(prepared)

        edge_program = to_edge(
            torch.export.export(qdq_module, sample_input, strict=True)
        )
        ep = edge_program.exported_program()
        gm = ep.graph_module

        gm = AnnotateQuantAttrs(ep)(gm).graph_module
        gm = FoldQDQ(ep)(gm).graph_module
        return gm, ep

    def test_insert_io_qdq_handles_dequant_encoding(self):
        """InsertIOQDQ should not KeyError when a node with a dequantize
        encoding feeds the output node (e.g. pre-quantized LLM parameters)."""
        gm, ep = self._build_quantized_graph()

        # Wire b__frozen_param0 (which has dequantize encoding) to output,
        # simulating the LLM topology from github issue #17732.
        param_node = None
        output_node = None
        for n in gm.graph.nodes:
            if n.name == "b__frozen_param0":
                param_node = n
            if n.op == "output":
                output_node = n

        self.assertIsNotNone(param_node)
        old_args = output_node.args[0]
        output_node.args = (
            ((old_args,) if not isinstance(old_args, tuple) else old_args)
            + (param_node,),
        )
        gm.graph.lint()
        gm.recompile()

        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)

        dq_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and hasattr(n.target, "__name__")
            and "dequantize" in n.target.__name__
            and any(u.op == "output" for u in n.users.keys())
        ]
        self.assertGreaterEqual(len(dq_nodes), 1)

    def test_insert_io_qdq_no_revisit(self):
        """InsertIOQDQ must not revisit newly inserted nodes."""
        gm, ep = self._build_quantized_graph()

        node_count_before = len(list(gm.graph.nodes))
        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)
        node_count_after = len(list(gm.graph.nodes))

        # AddModule with one input and one output should insert exactly
        # one quantize (input) and one dequantize (output) = +2 nodes.
        self.assertEqual(node_count_after, node_count_before + 2)

    def test_insert_requantize_for_mismatched_cat_inputs(self):
        class CatRequiresRequant(torch.nn.Module):
            def forward(self, x):
                first = torch.clamp(x, -0.1, 0.1)
                second = x * 10.0
                return torch.cat((first, second), dim=1)

        sample_input = (torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4),)
        gm, _ = self._build_quantized_graph(CatRequiresRequant(), sample_input)
        gm = InsertRequantize()(gm).graph_module

        cat_node = next(
            n for n in gm.graph.nodes if n.target == exir_ops.edge.aten.cat.default
        )
        cat_inputs = cat_node.args[0]
        to_copy_target = exir_ops.edge.aten._to_copy.default

        self.assertNotEqual(cat_inputs[0].target, to_copy_target)
        self.assertEqual(cat_inputs[1].target, to_copy_target)

    def test_cat_annotation_only_shares_output_with_first_input(self):
        class CatModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat((x, y), dim=1)

        sample_input = (
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
        )
        exported = torch.export.export(
            CatModule().eval(), sample_input, strict=True
        ).module()
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
        prepared = prepare_pt2e(exported, quantizer)

        cat_node = next(
            n for n in prepared.graph.nodes if n.target == torch.ops.aten.cat.default
        )
        second_input_node = cat_node.args[0][1]
        if second_input_node not in cat_node.meta[Q_ANNOTATION_KEY].input_qspec_map:
            second_input_node = second_input_node.args[0]

        self.assertIsInstance(
            cat_node.meta[Q_ANNOTATION_KEY].output_qspec,
            SharedQuantizationSpec,
        )
        self.assertNotIsInstance(
            cat_node.meta[Q_ANNOTATION_KEY].input_qspec_map[second_input_node],
            SharedQuantizationSpec,
        )

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

        try:
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module,
                sample_input,
                compiler_spec,
                generate_etrecord=True,
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise
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
