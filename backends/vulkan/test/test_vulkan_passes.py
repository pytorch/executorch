import unittest
from typing import Optional, Tuple

import torch

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.vulkan._passes import FuseQuantizedOpsTransform
from executorch.backends.vulkan._passes.fuse_patterns import FusePatternsPass

from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config,
    VulkanQuantizer,
)

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
    def test_fuse_int8pack_mm(self):
        K = 256
        N = 256
        model = SingleLinearModule(K, N)
        sample_inputs = model.get_sample_inputs()

        quantizer = VulkanQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_dynamic=False, weight_bits=8)
        )

        edge_manager = quantize_and_lower_module(
            model,
            sample_inputs,
            quantizer,
        )

        ep = edge_manager._edge_programs["forward"]
        edge_manager.transform(
            [
                AddmmToLinearTransform(),
                FuseQuantizedOpsTransform(ep),
            ]
        )

        gm = ep.graph_module

        self.assertEqual(op_node_count(gm, "_weight_int8pack_mm.default"), 1)
        self.assertEqual(op_node_count(gm, "dequantize_per_channel.default"), 0)

    def test_fuse_linear_qcs4w(self):
        K = 256
        N = 256
        model = SingleLinearModule(K, N)
        sample_inputs = model.get_sample_inputs()

        quantizer = VulkanQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_dynamic=False, weight_bits=4)
        )

        edge_manager = quantize_and_lower_module(
            model,
            sample_inputs,
            quantizer,
        )

        ep = edge_manager._edge_programs["forward"]
        edge_manager.transform(
            [
                AddmmToLinearTransform(),
                FuseQuantizedOpsTransform(ep),
            ]
        )

        gm = ep.graph_module

        self.assertEqual(op_node_count(gm, "linear_qcs4w.default"), 1)
        self.assertEqual(op_node_count(gm, "dequantize_per_channel.default"), 0)

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
        rotary_pass = FusePatternsPass(ep)
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
