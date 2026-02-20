import unittest
from typing import Optional, Tuple

import torch

from executorch.backends.vulkan._passes.fuse_patterns import FusePatternsPass

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

###################
## Common Models ##
###################


class SingleLinearModule(torch.nn.Module):
    def __init__(self, K=256, N=128):
        super().__init__()
        self.K = K
        self.N = N
        self.linear = torch.nn.Linear(K, N, bias=False)

    def forward(self, x):
        return self.linear(x)

    def get_sample_inputs(self):
        sample_inputs = (torch.rand(size=(32, self.K), dtype=torch.float32),)
        return sample_inputs


###########
## Tests ##
###########


def quantize_and_lower_module(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    quantizer: Quantizer,
    dynamic_shapes=None,
) -> EdgeProgramManager:
    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # TODO(T182928844): Delegate dim order op to backend.
        _check_ir_validity=False,
    )

    program = torch.export.export(
        model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()

    program = prepare_pt2e(program, quantizer)  # pyre-ignore
    # Calibrate
    program(*sample_inputs)

    program = convert_pt2e(program)

    program = torch.export.export(program, sample_inputs, dynamic_shapes=dynamic_shapes)

    edge_program = to_edge(
        program,
        compile_config=edge_compile_config,
    )

    return edge_program


def get_target_canonical_name(node: torch.fx.Node) -> Optional[str]:
    if node.op != "call_function":
        return None
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name


def op_node_count(graph_module: torch.fx.GraphModule, canonical_op_name: str) -> int:
    count = 0
    for node in graph_module.graph.nodes:
        canonical_name = get_target_canonical_name(node)
        if canonical_name is not None and canonical_name == canonical_op_name:
            count += 1
    return count


class TestVulkanPasses(unittest.TestCase):
    def test_fuse_rotary_emb(self):
        """Test conversion of rotary embedding pattern to et_vk.apply_rotary_emb custom op."""

        class RotaryEmbeddingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                xq: torch.Tensor,
                xk: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                # This implementation matches the apply_rotary_emb function in rope.py
                # Split into real and imaginary parts
                xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
                xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

                # Reshape frequencies for broadcasting
                freqs_cos = self._reshape_for_broadcast(freqs_cos, xq_r)
                freqs_sin = self._reshape_for_broadcast(freqs_sin, xq_r)

                # Apply rotary embedding
                xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
                xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
                xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
                xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

                # Recombine real and imaginary parts
                xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
                xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

                return xq_out.type_as(xq), xk_out.type_as(xk)

            def _reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
                """Helper function to reshape frequencies for broadcasting"""
                ndim = x.ndim
                freqs_cis_ndim = freqs_cis.ndim
                if freqs_cis_ndim == 3:
                    # freqs_cis: (seq_len, n_heads, head_dim // 2)
                    shape = [
                        d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
                        for i, d in enumerate(x.shape)
                    ]
                else:
                    # freqs_cis: (seq_len, head_dim // 2)
                    shape = [
                        d if i == 1 or i == ndim - 1 else 1
                        for i, d in enumerate(x.shape)
                    ]
                return freqs_cis.view(shape)

        # Create sample inputs based on the test file
        batch_size = 1
        seq_len = 5
        n_heads = 32
        n_kv_heads = 8
        head_dim = 2048

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float)
        xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float)
        freqs_cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float)
        freqs_sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float)

        sample_inputs = (xq, xk, freqs_cos, freqs_sin)

        model = RotaryEmbeddingModel()

        # Export the model
        edge_compile_config = EdgeCompileConfig(
            _skip_dim_order=False,
            _check_ir_validity=False,
        )

        program = torch.export.export(model, sample_inputs, strict=True)

        edge_manager = to_edge(
            program,
            compile_config=edge_compile_config,
        )

        # Apply the rotary embedding pass
        ep = edge_manager._edge_programs["forward"]
        rotary_pass = FusePatternsPass()
        rotary_pass._exported_program = ep
        result = rotary_pass.call(ep.graph_module)

        # Verify that the pass was successful
        self.assertTrue(result.modified)

        # Check that the custom op was created
        gm = ep.graph_module
        custom_op_count = 0
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and hasattr(node.target, "__name__")
                and "apply_rotary_emb" in str(node.target)
            ):
                custom_op_count += 1

        # We expect at least one custom op to be created
        self.assertGreater(custom_op_count, 0)

    def test_fuse_q8ta_linear(self):
        """Test that sequential quantized linears fuse into q8ta_linear when output quantization is present."""
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )

        class TwoLinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(128, 64, bias=False)
                self.linear2 = torch.nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TwoLinearModule()
        sample_inputs = (torch.randn(4, 128),)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        edge_program = quantize_and_lower_module(model, sample_inputs, quantizer)

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        # The first linear should fuse to q8ta_linear (has output quantization
        # from the second linear's input quantize node)
        q8ta_linear_count = op_node_count(gm, "q8ta_linear.default")
        self.assertGreaterEqual(
            q8ta_linear_count,
            1,
            "Expected at least one q8ta_linear op from output-quantized linear fusion",
        )

    def test_fuse_q8ta_linear_gemv(self):
        """Test that batch-1 quantized linear fuses into q8ta_linear_gemv."""
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )

        class TwoLinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(128, 64, bias=False)
                self.linear2 = torch.nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TwoLinearModule()
        # Batch size 1 to trigger gemv variant
        sample_inputs = (torch.randn(1, 128),)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        edge_program = quantize_and_lower_module(model, sample_inputs, quantizer)

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        # With batch size 1, the first linear should fuse to q8ta_linear_gemv
        q8ta_linear_gemv_count = op_node_count(gm, "q8ta_linear_gemv.default")
        self.assertGreaterEqual(
            q8ta_linear_gemv_count,
            1,
            "Expected at least one q8ta_linear_gemv op for batch-1 linear fusion",
        )

    def test_fuse_three_chained_q8ta_linears(self):
        """Test that 3 consecutive quantized linears fuse into q8ta_linear ops with
        correct quant params at each layer boundary.

        Each linear's input scale/zp (args[1], args[2]) must equal its predecessor's
        output scale/zp (args[6], args[7]). This is a regression test for a bug where
        topological pattern replacement caused later linears to read scale/zp from the
        wrong arg position of the already-replaced q8ta_linear node, producing wildly
        incorrect quantization parameters (outputs saturating to -128/127).
        """
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )

        class ThreeLinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(256, 128, bias=False)
                self.linear2 = torch.nn.Linear(128, 64, bias=False)
                self.linear3 = torch.nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.linear3(self.linear2(self.linear1(x)))

        model = ThreeLinearModule()
        # Batch size 4 to select q8ta_linear (not the gemv variant)
        sample_inputs = (torch.randn(4, 256),)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        edge_program = quantize_and_lower_module(model, sample_inputs, quantizer)

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        q8ta_nodes = [
            node
            for node in gm.graph.nodes
            if get_target_canonical_name(node) == "q8ta_linear.default"
        ]
        self.assertGreaterEqual(
            len(q8ta_nodes),
            2,
            "Expected at least 2 q8ta_linear ops from 3 chained quantized linears",
        )

        # For each consecutive q8ta_linear pair, the boundary scale/zp must be
        # consistent: linear_i.output_scale == linear_{i+1}.input_scale.
        # Before the fix, linear_{i+1}.input_scale was incorrectly read from the
        # replaced q8ta_linear node's input args instead of the dq node's args.
        for i in range(len(q8ta_nodes) - 1):
            self.assertEqual(
                q8ta_nodes[i].args[6],
                q8ta_nodes[i + 1].args[1],
                f"q8ta_linear[{i}].output_scale should equal q8ta_linear[{i + 1}].input_scale",
            )
            self.assertEqual(
                q8ta_nodes[i].args[7],
                q8ta_nodes[i + 1].args[2],
                f"q8ta_linear[{i}].output_zero_point should equal q8ta_linear[{i + 1}].input_zero_point",
            )

    def test_fuse_q8ta_linear_gemv_non_aligned_oc(self):
        """Test that quantized linear with non-aligned output channels (not multiple of 4) fuses correctly."""
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )

        class TwoLinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Use non-aligned output channels (9 is not a multiple of 4)
                self.linear1 = torch.nn.Linear(128, 9, bias=False)
                self.linear2 = torch.nn.Linear(9, 4, bias=False)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TwoLinearModule()
        sample_inputs = (torch.randn(1, 128),)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        edge_program = quantize_and_lower_module(model, sample_inputs, quantizer)

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        # The first linear (OC=9, not multiple of 4) should still fuse
        q8ta_linear_gemv_count = op_node_count(gm, "q8ta_linear_gemv.default")
        self.assertGreaterEqual(
            q8ta_linear_gemv_count,
            1,
            "Expected non-aligned OC linear to fuse into q8ta_linear_gemv",
        )
