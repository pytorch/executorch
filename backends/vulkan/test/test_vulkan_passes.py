import operator
import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import torch

from executorch.backends.vulkan._passes.fuse_patterns import FusePatternsPass
from executorch.backends.vulkan._passes.squeeze_unsqueeze_inputs import (
    SqueezeUnsqueezeInputs,
)
from executorch.backends.vulkan.patterns.quantized_linear import (
    find_quantized_linear_patterns,
    replace_quantized_linear_patterns,
)
from executorch.backends.vulkan.patterns.sdpa import (
    CausalSDPAMatch,
    is_custom_sdpa_node,
    is_sdpa_with_kv_cache_node,
    is_update_cache_node,
)

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)

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


class HfRopeFullTableModule(torch.nn.Module):
    def __init__(self, broken_k_arm: bool = False, sin_start: int = 2) -> None:
        super().__init__()
        self.broken_k_arm = broken_k_arm
        self.sin_start = sin_start

    def forward(self, xq, xk, cos_table, sin_table):
        cos = cos_table[2:3].unsqueeze(1)
        sin = sin_table[self.sin_start : self.sin_start + 1].unsqueeze(1)
        q1 = xq[..., : xq.shape[-1] // 2]
        q2 = xq[..., xq.shape[-1] // 2 :]
        k1 = xk[..., : xk.shape[-1] // 2]
        k2 = xk[..., xk.shape[-1] // 2 :]
        q = (xq * cos) + (torch.cat((-q2, q1), dim=-1) * sin)
        k_rotation = torch.cat((-k2, k1), dim=-1) * sin
        if self.broken_k_arm:
            return q, (xk * cos) - k_rotation
        return q, (xk * cos) + k_rotation

    def get_sample_inputs(self):
        return (
            torch.rand(1, 1, 4, 32),
            torch.rand(1, 1, 2, 32),
            torch.rand(8, 32),
            torch.rand(8, 32),
        )


class TiedLinearPair(torch.nn.Module):
    def __init__(self, in_features: int = 256, out_features: int = 128) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.linear2 = torch.nn.Linear(in_features, out_features)
        self.linear2.weight = self.linear1.weight

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)

    def get_sample_inputs(self):
        return (torch.rand(8, self.in_features),)


class RmsNormVariantModule(torch.nn.Module):
    def __init__(self, exponent: Optional[float]) -> None:
        super().__init__()
        self.exponent = exponent
        self.weight = torch.nn.Parameter(torch.ones(64))

    def forward(self, x):
        mean_sq = x.pow(2).mean(-1, keepdim=True) + 1e-6
        if self.exponent is None:
            rstd = torch.rsqrt(mean_sq)
        else:
            rstd = torch.pow(mean_sq, self.exponent)
        return (x * rstd) * self.weight

    def get_sample_inputs(self):
        return (torch.rand(2, 8, 64),)


class SelectAsSymIntModule(torch.nn.Module):
    def __init__(self, bounded: bool = True, use_select: bool = True) -> None:
        super().__init__()
        self.bounded = bounded
        self.use_select = use_select

    def forward(self, index, x):
        if self.use_select:
            value = index.select(0, 0).item()
        else:
            value = index.sum().item()
        if not self.bounded:
            return x.sum() + value
        torch._check(value >= 3)
        torch._check(value <= 17)
        return x[:, :value].sum()

    def get_sample_inputs(self):
        return (torch.tensor([5], dtype=torch.int64), torch.rand(2, 32))


class CausalSdpaModule(torch.nn.Module):
    def __init__(self, update_key: bool, update_value: bool) -> None:
        super().__init__()
        self.update_key = update_key
        self.update_value = update_value
        self.register_buffer("key_cache", torch.zeros(1, 8, 2, 4))
        self.register_buffer("value_cache", torch.zeros(1, 8, 2, 4))

    def forward(self, query, key, value):
        if self.update_key:
            torch.ops.llama.update_cache(key, self.key_cache, 0)
        if self.update_value:
            torch.ops.llama.update_cache(value, self.value_cache, 0)
        return torch.ops.llama.custom_sdpa(
            query,
            self.key_cache,
            self.value_cache,
            0,
            None,
            0.0,
            True,
            None,
        )

    def get_sample_inputs(self):
        return (
            torch.rand(1, 1, 2, 4),
            torch.rand(1, 1, 2, 4),
            torch.rand(1, 1, 2, 4),
        )


class SharedCacheCausalSdpaModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 8, 2, 4))

    def forward(self, query, projected_cache_value):
        torch.ops.llama.update_cache(projected_cache_value, self.cache, 0)
        return torch.ops.llama.custom_sdpa(
            query,
            self.cache,
            self.cache,
            0,
            None,
            0.0,
            True,
            None,
        )

    def get_sample_inputs(self):
        return (
            torch.rand(1, 1, 2, 4),
            torch.rand(1, 1, 2, 4),
        )


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


def lower_module(model: torch.nn.Module):
    program = torch.export.export(model, model.get_sample_inputs(), strict=True)
    return to_edge(
        program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def run_fuse_patterns(exported_program):
    fuse_pass = FusePatternsPass()
    fuse_pass._exported_program = exported_program
    return fuse_pass.call(exported_program.graph_module)


def nodes_named(graph_module: torch.fx.GraphModule, canonical_op_name: str):
    return [
        node
        for node in graph_module.graph.nodes
        if get_target_canonical_name(node) == canonical_op_name
    ]


def quantized_linear_matches(exported_program):
    matches = []
    for node in exported_program.graph_module.graph.nodes:
        match = find_quantized_linear_patterns(node)
        if match is not None:
            matches.append(match)
    return matches


def quantize_tied_pair(model: TiedLinearPair, config) -> TiedLinearPair:
    quantize_(model, config)
    model.linear2.weight = model.linear1.weight
    return model


def shared_8da4w_program(
    in_features: int = 256,
    out_features: int = 128,
    group_size: int = 32,
    weight_dtype=torch.int4,
):
    model = TiedLinearPair(in_features, out_features).eval()
    config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=weight_dtype,
        weight_granularity=PerGroup(group_size),
    )
    return lower_module(quantize_tied_pair(model, config))


def causal_sdpa_program(update_key: bool, update_value: bool):
    exported_program = lower_module(CausalSdpaModule(update_key, update_value))
    return unsafe_remove_auto_functionalized_pass(exported_program)


def shared_cache_causal_sdpa_program():
    exported_program = lower_module(SharedCacheCausalSdpaModule())
    return unsafe_remove_auto_functionalized_pass(exported_program)


class TestVulkanPasses(unittest.TestCase):
    def test_squeeze_preserves_gelu_kwargs_only_on_gelu(self):
        input_arg = MagicMock()
        input_arg.node.meta = {"val": torch.empty(2, 1, 4)}
        kwargs = {"approximate": "tanh"}
        metadata = {"val": torch.empty(2, 1, 4)}

        with patch.object(ExportPass, "call_operator", autospec=True) as base_call:
            base_call.side_effect = [MagicMock(), MagicMock(), MagicMock()]
            SqueezeUnsqueezeInputs().call_operator(
                exir_ops.edge.aten.gelu.default,
                (input_arg,),
                kwargs,
                metadata,
            )

        calls = base_call.call_args_list
        self.assertEqual(calls[0].args[3], {})
        self.assertEqual(calls[1].args[3], kwargs)
        self.assertEqual(calls[2].args[3], {})

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


class GemmaMatcherWitnessTest(unittest.TestCase):
    def test_full_table_hf_rope_preserves_tables_start_and_meta_shapes(self):
        model = HfRopeFullTableModule()
        exported_program = lower_module(model)

        result = run_fuse_patterns(exported_program)
        fused_nodes = nodes_named(result.graph_module, "apply_rotary_emb_hf.default")
        self.assertEqual(len(fused_nodes), 1)
        fused = fused_nodes[0]
        self.assertEqual(fused.args[2].target, "cos_table")
        self.assertEqual(fused.args[3].target, "sin_table")
        self.assertEqual(fused.args[4], 2)

        output_shapes = {
            node.args[1]: tuple(node.meta["val"].shape)
            for node in result.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target is operator.getitem
            and node.args[0] is fused
        }
        self.assertEqual(output_shapes, {0: (1, 1, 4, 32), 1: (1, 1, 2, 32)})

    def test_full_table_hf_rope_rejects_changed_k_arithmetic(self):
        result = run_fuse_patterns(lower_module(HfRopeFullTableModule(True)))

        self.assertFalse(result.modified)
        self.assertEqual(
            nodes_named(result.graph_module, "apply_rotary_emb_hf.default"), []
        )

    def test_full_table_hf_rope_rejects_mismatched_frequency_slices(self):
        result = run_fuse_patterns(lower_module(HfRopeFullTableModule(sin_start=3)))

        self.assertEqual(
            nodes_named(result.graph_module, "apply_rotary_emb_hf.default"), []
        )

    def test_shared_8da4w_reuses_only_weight_storage(self):
        exported_program = shared_8da4w_program()
        matches = quantized_linear_matches(exported_program)
        self.assertEqual(len(matches), 2)
        self.assertIs(matches[0].weight_node, matches[1].weight_node)

        result = run_fuse_patterns(exported_program)
        fused = nodes_named(result.graph_module, "linear_dq8ca_q4gsw.default")
        self.assertEqual(len(fused), 2)
        for argument_index in (3, 4, 5):
            self.assertIs(fused[0].args[argument_index], fused[1].args[argument_index])
        self.assertIsNot(fused[0].args[7], fused[1].args[7])
        self.assertEqual(
            exported_program._et_vk_param_modification_tags,
            {
                "linear1.parametrizations.weight.original0": "4 bit linear weight",
                "linear1.parametrizations.weight.original1": "4 bit linear scales",
            },
        )

    def test_shared_8da4w_rejects_incomplete_mutation_tags(self):
        exported_program = shared_8da4w_program()
        matches = quantized_linear_matches(exported_program)
        self.assertEqual(len(matches), 2)
        replace_quantized_linear_patterns(
            exported_program, exported_program.graph_module, matches[0]
        )
        tags = exported_program._et_vk_param_modification_tags
        scales_name = next(
            name for name, tag in tags.items() if tag == "4 bit linear scales"
        )
        del tags[scales_name]

        with self.assertRaisesRegex(
            RuntimeError, "shared quantized linear mutation tags are inconsistent"
        ):
            replace_quantized_linear_patterns(
                exported_program, exported_program.graph_module, matches[1]
            )

    def test_shared_weight_only_4bit_linear_is_rejected_fail_closed(self):
        model = TiedLinearPair().eval()
        config = IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(32),
        )
        exported_program = lower_module(quantize_tied_pair(model, config))

        with self.assertRaisesRegex(
            RuntimeError, "shared quantized linear mutation tags are inconsistent"
        ):
            run_fuse_patterns(exported_program)

    def test_shared_8da4w_rejects_packed_weight_geometry(self):
        exported_program = shared_8da4w_program(in_features=252, group_size=4)

        with self.assertRaisesRegex(
            RuntimeError, "shared 8da4w packed weight geometry mismatch"
        ):
            run_fuse_patterns(exported_program)

    def test_shared_8da4w_rejects_weight_sums_geometry(self):
        for out_features, error in (
            (130, "shared 8da4w packed weight geometry mismatch"),
            (132, "shared 8da4w weight sums geometry mismatch"),
        ):
            with self.subTest(out_features=out_features, error=error):
                exported_program = shared_8da4w_program(out_features=out_features)

                with self.assertRaisesRegex(RuntimeError, error):
                    run_fuse_patterns(exported_program)

    def test_shared_8da4w_rejects_unexpected_dequant_abi(self):
        exported_program = shared_8da4w_program(weight_dtype=torch.int2)

        with self.assertRaisesRegex(
            RuntimeError, "shared 8da4w weight has an unexpected dequant ABI"
        ):
            run_fuse_patterns(exported_program)

    def test_rms_norm_accepts_rsqrt_and_pow_negative_half(self):
        for exponent, expected_rsqrt, expected_pow in (
            (None, 1, 1),
            (-0.5, 0, 2),
        ):
            with self.subTest(exponent=exponent):
                exported_program = lower_module(RmsNormVariantModule(exponent))
                self.assertEqual(
                    op_node_count(exported_program.graph_module, "rsqrt.default"),
                    expected_rsqrt,
                )
                self.assertEqual(
                    op_node_count(exported_program.graph_module, "pow.Tensor_Scalar"),
                    expected_pow,
                )
                result = run_fuse_patterns(exported_program)
                self.assertEqual(
                    op_node_count(result.graph_module, "rms_norm.default"), 1
                )

    def test_rms_norm_rejects_other_reciprocal_exponents(self):
        for exponent in (-1.0, -0.25):
            with self.subTest(exponent=exponent):
                exported_program = lower_module(RmsNormVariantModule(exponent))
                self.assertEqual(
                    op_node_count(exported_program.graph_module, "pow.Tensor_Scalar"),
                    2,
                )
                result = run_fuse_patterns(exported_program)
                self.assertFalse(result.modified)
                self.assertEqual(
                    op_node_count(result.graph_module, "rms_norm.default"), 0
                )

    def test_select_as_symint_preserves_bounded_and_unbounded_ranges(self):
        for bounded, expected_range in ((True, (3, 17)), (False, (None, None))):
            with self.subTest(bounded=bounded):
                result = run_fuse_patterns(
                    lower_module(SelectAsSymIntModule(bounded=bounded))
                )
                selected = nodes_named(result.graph_module, "select_as_symint.default")
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0].meta["et_vk_value_range"], expected_range)
                constraints = [
                    node
                    for node in result.graph_module.graph.nodes
                    if "sym_constrain_range.default" in str(node.target)
                ]
                self.assertEqual(len(constraints), 1)
                self.assertEqual(
                    constraints[0].kwargs,
                    {"min": expected_range[0], "max": expected_range[1]},
                )

    def test_select_as_symint_rejects_non_select_scalar(self):
        result = run_fuse_patterns(lower_module(SelectAsSymIntModule(use_select=False)))

        self.assertFalse(result.modified)
        self.assertEqual(
            nodes_named(result.graph_module, "select_as_symint.default"), []
        )

    def test_select_as_symint_eager_requires_integral_input(self):
        self.assertEqual(
            torch.ops.et_vk.select_as_symint.default(
                torch.tensor([5, 6], dtype=torch.int64), 0, 1
            ),
            6,
        )
        with self.assertRaisesRegex(ValueError, "requires an integral input"):
            torch.ops.et_vk.select_as_symint.default(torch.tensor([5.0, 6.0]), 0, 1)

    def test_causal_sdpa_requires_both_cache_updates(self):
        for update_key, update_value, expected in (
            (True, True, (True, True, True)),
            (True, False, (True, False, False)),
            (False, True, (False, True, False)),
            (False, False, (False, False, False)),
        ):
            with self.subTest(update_key=update_key, update_value=update_value):
                exported_program = causal_sdpa_program(update_key, update_value)
                sdpa_node = next(
                    node
                    for node in exported_program.graph_module.graph.nodes
                    if is_custom_sdpa_node(node)
                )
                match = CausalSDPAMatch(sdpa_node)
                actual = (
                    match.update_key_cache_node is not None,
                    match.update_value_cache_node is not None,
                    match.match_found,
                )
                self.assertEqual(actual, expected)
                if match.match_found:
                    self.assertEqual(match.query_node.target, "query")
                    self.assertEqual(match.key_projection_node.target, "key")
                    self.assertEqual(match.value_projection_node.target, "value")

    def test_causal_sdpa_replaces_complete_cache_topology(self):
        exported_program = causal_sdpa_program(True, True)
        result = run_fuse_patterns(exported_program)

        self.assertTrue(result.modified)
        self.assertEqual(
            sum(is_update_cache_node(node) for node in result.graph_module.graph.nodes),
            0,
        )
        self.assertEqual(
            sum(
                is_sdpa_with_kv_cache_node(node)
                for node in result.graph_module.graph.nodes
            ),
            1,
        )

    def test_causal_sdpa_rejects_shared_key_value_cache_update(self):
        exported_program = shared_cache_causal_sdpa_program()
        sdpa_node = next(
            node
            for node in exported_program.graph_module.graph.nodes
            if is_custom_sdpa_node(node)
        )
        match = CausalSDPAMatch(sdpa_node)

        self.assertIs(match.update_key_cache_node, match.update_value_cache_node)
        self.assertFalse(match.match_found)

        result = run_fuse_patterns(exported_program)
        self.assertFalse(result.modified)
        self.assertEqual(
            sum(is_update_cache_node(node) for node in result.graph_module.graph.nodes),
            1,
        )
        self.assertEqual(
            sum(is_custom_sdpa_node(node) for node in result.graph_module.graph.nodes),
            1,
        )
        self.assertEqual(
            sum(
                is_sdpa_with_kv_cache_node(node)
                for node in result.graph_module.graph.nodes
            ),
            0,
        )

    def test_causal_sdpa_rejects_incomplete_optional_arguments(self):
        for argument_count in (4, 5, 6):
            with self.subTest(argument_count=argument_count):
                graph = torch.fx.Graph()
                query = graph.placeholder("query")
                key = graph.placeholder("key")
                value = graph.placeholder("value")
                full_args = (query, key, value, 0, None, 0.0)
                sdpa_node = graph.call_function(
                    torch.ops.llama.custom_sdpa.default,
                    full_args[:argument_count],
                )

                self.assertFalse(CausalSDPAMatch(sdpa_node).match_found)
