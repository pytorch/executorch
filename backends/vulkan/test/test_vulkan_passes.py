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
    def test_fuse_torchao_quantized_embedding(self):
        """A torchao-dialect 4-bit weight-only quantized embedding
        (torchao.dequantize_affine -> aten.embedding) should fuse into a single
        et_vk.embedding_q4gsw.default node, with the dequant_affine and embedding
        nodes removed.
        """
        import executorch.backends.vulkan.custom_ops_lib  # noqa: registers et_vk ops
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.utils import unwrap_tensor_subclass

        vocab_size = 64
        embed_dim = 128
        group_size = 32

        class EmbModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab_size, embed_dim)

            def forward(self, x):
                return self.emb(x)

        model = EmbModule()
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(group_size)
            ),
            filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Embedding),
        )
        unwrap_tensor_subclass(model)

        sample_inputs = (torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),)
        # Eager reference output of the quantized embedding, before any fusion.
        eager_ref = model(*sample_inputs)

        program = torch.export.export(model, sample_inputs, strict=True)
        edge_program = to_edge(
            program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        self.assertEqual(op_node_count(gm, "embedding_q4gsw.default"), 1)
        self.assertEqual(op_node_count(gm, "dequantize_affine.default"), 0)
        self.assertEqual(op_node_count(gm, "embedding.default"), 0)

        # Verify the fused op carries the expected args:
        # (weight, scales, group_size, indices, is_linear_weight)
        fused_node = next(
            n
            for n in gm.graph.nodes
            if get_target_canonical_name(n) == "embedding_q4gsw.default"
        )
        self.assertEqual(fused_node.args[2], group_size)
        # The weight is always packed in the LINEAR-weight q4gsw layout so a tied
        # embedding/LM-head weight is supported, so is_linear_weight is True.
        self.assertTrue(fused_node.args[4])

        # The weight placeholder is repacked from unpacked int8 [vocab, embed_dim]
        # to linear-convention 4-bit packed uint8. embed_dim % 32 == 0 means
        # embed_dim / 2 is a multiple of 16, so the linear packing's mult-of-8
        # inner-dim padding is inert and the packed inner dim stays embed_dim / 2.
        weight_node = fused_node.args[0]
        self.assertEqual(weight_node.meta["val"].dtype, torch.uint8)
        self.assertEqual(
            tuple(weight_node.meta["val"].shape), (vocab_size, embed_dim // 2)
        )

        # Numerically verify the fused op (via its CompositeExplicitAutograd
        # reference impl) reproduces the eager quantized embedding output. This
        # exercises the repacked weight + scale layout end-to-end against an
        # independently-computed reference.
        from executorch.backends.transforms.utils import get_param_tensor

        packed_weight = get_param_tensor(ep, weight_node)
        scales_tensor = get_param_tensor(ep, fused_node.args[1])
        fused_out = torch.ops.et_vk.embedding_q4gsw.default(
            packed_weight,
            scales_tensor,
            group_size,
            sample_inputs[0],
            True,
        )
        self.assertTrue(torch.allclose(fused_out, eager_ref, atol=1e-3, rtol=1e-3))

    def test_torchao_quantized_embedding_rejects_bad_embed_dim(self):
        """A torchao 4-bit quantized embedding whose embed_dim is not a multiple
        of 32 must NOT fuse: the runtime shader asserts embed_dim % 32 == 0
        (VK_CHECK in EmbeddingQ4gsw.cpp), so the matcher's input-validation guard
        rejects it and the op falls back to CPU rather than producing an op the
        runtime would abort on. embed_dim=48 is divisible by group_size=16 (so the
        group-size, zero_point, and qmin/qmax guards all pass) but 48 % 32 != 0.
        """
        import executorch.backends.vulkan.custom_ops_lib  # noqa: registers et_vk ops
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.utils import unwrap_tensor_subclass

        vocab_size = 64
        embed_dim = 48  # not a multiple of 32
        group_size = 16

        class EmbModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab_size, embed_dim)

            def forward(self, x):
                return self.emb(x)

        model = EmbModule()
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(group_size)
            ),
            filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Embedding),
        )
        unwrap_tensor_subclass(model)

        sample_inputs = (torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),)

        program = torch.export.export(model, sample_inputs, strict=True)
        edge_program = to_edge(
            program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        fuse_pass.call(ep.graph_module)

        gm = ep.graph_module

        # The guard rejected the match: no fused op, and the original
        # aten.embedding lookup remains for the CPU fallback path.
        self.assertEqual(op_node_count(gm, "embedding_q4gsw.default"), 0)
        self.assertEqual(op_node_count(gm, "embedding.default"), 1)

    def test_fuse_torchao_quantized_embedding_shared_weight(self):
        """A single torchao-quantized embedding weight shared by multiple
        aten.embedding call sites (two dequantize_affine -> embedding chains over
        the same weight placeholder) must fuse into two et_vk.embedding_q4gsw
        nodes that reference the SAME repacked weight, and the weight must only be
        repacked once (regression test: repacking the shared state-dict entry
        twice would corrupt it, halving its width on the second pass).
        """
        import executorch.backends.vulkan.custom_ops_lib  # noqa: registers et_vk ops
        from executorch.backends.transforms.utils import get_param_tensor
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.utils import unwrap_tensor_subclass

        vocab_size = 64
        embed_dim = 128
        group_size = 32

        class TwoLookupEmbModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab_size, embed_dim)

            def forward(self, x, y):
                # Two lookups into the same embedding table.
                return self.emb(x) + self.emb(y)

        model = TwoLookupEmbModule()
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(group_size)
            ),
            filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Embedding),
        )
        unwrap_tensor_subclass(model)

        sample_inputs = (
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            torch.tensor([5, 6, 7, 8, 9], dtype=torch.int64),
        )
        eager_ref = model(*sample_inputs)

        program = torch.export.export(model, sample_inputs, strict=True)
        edge_program = to_edge(
            program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module

        # Both embedding call sites should fuse; neither dequant_affine nor
        # embedding nodes should remain.
        self.assertEqual(op_node_count(gm, "embedding_q4gsw.default"), 2)
        self.assertEqual(op_node_count(gm, "dequantize_affine.default"), 0)
        self.assertEqual(op_node_count(gm, "embedding.default"), 0)

        fused_nodes = [
            n
            for n in gm.graph.nodes
            if get_target_canonical_name(n) == "embedding_q4gsw.default"
        ]
        # Both fused nodes must reference the same (single) repacked weight.
        self.assertEqual(fused_nodes[0].args[0], fused_nodes[1].args[0])

        # Both fused nodes use the linear-weight q4gsw layout (is_linear_weight).
        self.assertTrue(fused_nodes[0].args[4])
        self.assertTrue(fused_nodes[1].args[4])

        # The shared weight must be repacked exactly once: linear-convention
        # 4-bit packed uint8. Since embed_dim % 32 == 0, embed_dim / 2 is a
        # multiple of 16 so the linear packing's inner-dim padding is inert and
        # the packed inner dim is embed_dim / 2. A double-pack would yield
        # [vocab, embed_dim / 4].
        weight_node = fused_nodes[0].args[0]
        packed_weight = get_param_tensor(ep, weight_node)
        self.assertEqual(packed_weight.dtype, torch.uint8)
        self.assertEqual(tuple(packed_weight.shape), (vocab_size, embed_dim // 2))

        # End-to-end numerical check against the eager reference.
        scales_tensor = get_param_tensor(ep, fused_nodes[0].args[1])
        emb_x = torch.ops.et_vk.embedding_q4gsw.default(
            packed_weight, scales_tensor, group_size, sample_inputs[0], True
        )
        emb_y = torch.ops.et_vk.embedding_q4gsw.default(
            packed_weight, scales_tensor, group_size, sample_inputs[1], True
        )
        self.assertTrue(torch.allclose(emb_x + emb_y, eager_ref, atol=1e-3, rtol=1e-3))

    def test_register_param_mutation(self):
        """utils.register_param_mutation is a storage-keyed idempotency guard:
        the first call for a param returns True (proceed and record the tag), a
        repeat with the same tag returns False (skip), and a call with a
        conflicting tag raises.
        """
        import executorch.backends.vulkan.utils as vk_utils

        model = SingleLinearModule()
        program = torch.export.export(model, model.get_sample_inputs(), strict=True)
        edge_program = to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        ep = edge_program._edge_programs["forward"]
        gm = ep.graph_module

        # Grab the linear weight parameter placeholder. The fused linear node
        # consumes it as a constant tensor arg.
        weight_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "placeholder" and vk_utils.is_param(ep, n)
        )

        # First call for this param: records the tag, returns True (proceed).
        self.assertTrue(vk_utils.register_param_mutation(ep, weight_node, "fmt_a"))
        # Repeat with the same tag: already mutated this way, returns False (skip).
        self.assertFalse(vk_utils.register_param_mutation(ep, weight_node, "fmt_a"))
        self.assertFalse(vk_utils.register_param_mutation(ep, weight_node, "fmt_a"))
        # Conflicting tag on the same param: an incompatible re-mutation, raises.
        with self.assertRaises(RuntimeError):
            vk_utils.register_param_mutation(ep, weight_node, "fmt_b")

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

    def test_fuse_quantized_pixel_shuffle(self):
        """An un-decomposed pixel_shuffle wrapped in dequantize/quantize_per_tensor
        ops should fuse into a single et_vk.q8ta_pixel_shuffle.default node, and
        none of the original quant/dequant nodes should remain.

        The matcher relies on the partitioner's `ops_to_not_decompose()` hook
        keeping `aten.pixel_shuffle.default` intact through edge lowering. We
        replicate that behaviour here via `EdgeCompileConfig.preserve_ops` so
        the test exercises the same graph shape that the partitioner produces
        end-to-end.
        """

        class PixelShuffleModule(torch.nn.Module):
            def forward(self, x):
                x_dq = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 0.1, 0, -128, 127, torch.int8
                )
                y = torch.nn.functional.pixel_shuffle(x_dq, 2)
                return torch.ops.quantized_decomposed.quantize_per_tensor(
                    y, 0.05, 1, -128, 127, torch.int8
                )

        # Use a non-square H/W and a W that is not a multiple of 4 so the
        # geometry checks exercise the same shapes the model uses.
        x = torch.randint(-128, 127, (1, 96, 16, 9), dtype=torch.int8)
        program = torch.export.export(PixelShuffleModule(), (x,), strict=True)
        edge_program = to_edge(
            program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                preserve_ops=[torch.ops.aten.pixel_shuffle.default],
            ),
        )

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        result = fuse_pass.call(ep.graph_module)

        self.assertTrue(result.modified)

        gm = ep.graph_module
        self.assertEqual(op_node_count(gm, "q8ta_pixel_shuffle.default"), 1)
        self.assertEqual(op_node_count(gm, "view_copy.default"), 0)
        self.assertEqual(op_node_count(gm, "permute_copy.default"), 0)
        self.assertEqual(op_node_count(gm, "pixel_shuffle.default"), 0)
        self.assertEqual(op_node_count(gm, "dequantize_per_tensor.default"), 0)
        self.assertEqual(op_node_count(gm, "quantize_per_tensor.default"), 0)

        # Verify the fused op carries the correct args.
        fused_node = next(
            n
            for n in gm.graph.nodes
            if get_target_canonical_name(n) == "q8ta_pixel_shuffle.default"
        )
        # args = (input, input_scale, input_zp, inv_output_scale, output_zp, r)
        self.assertEqual(fused_node.args[1], 0.1)
        self.assertEqual(fused_node.args[2], 0)
        # 1.0 / 0.05 == 20.0
        self.assertEqual(fused_node.args[3], 20.0)
        self.assertEqual(fused_node.args[4], 1)
        self.assertEqual(fused_node.args[5], 2)

    def test_quantized_pixel_shuffle_pattern_rejects_non_match(self):
        """A `dq -> relu -> q` chain (no pixel_shuffle in between) must NOT be
        fused. The new matcher only triggers when a single
        `aten.pixel_shuffle.default` node sits between the dequant/quant pair.
        """

        class NonPixelShuffleModule(torch.nn.Module):
            def forward(self, x):
                x_dq = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 0.1, 0, -128, 127, torch.int8
                )
                y = torch.nn.functional.relu(x_dq)
                return torch.ops.quantized_decomposed.quantize_per_tensor(
                    y, 0.1, 0, -128, 127, torch.int8
                )

        x = torch.randint(-128, 127, (1, 96, 16, 9), dtype=torch.int8)
        program = torch.export.export(NonPixelShuffleModule(), (x,), strict=True)
        edge_program = to_edge(
            program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                preserve_ops=[torch.ops.aten.pixel_shuffle.default],
            ),
        )

        ep = edge_program._edge_programs["forward"]
        fuse_pass = FusePatternsPass()
        fuse_pass._exported_program = ep
        fuse_pass.call(ep.graph_module)

        gm = ep.graph_module
        self.assertEqual(op_node_count(gm, "q8ta_pixel_shuffle.default"), 0)
