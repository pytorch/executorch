# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.vulkan.op_registry import has_impl, vulkan_supported_ops
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.examples.models.gemma4.webgpu_partitioner import (
    _extra_op_features,
    _webgpu_allowlist,
    build_webgpu_partitioner,
    Gemma4WebGPUPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class ExportPartitionersTest(unittest.TestCase):
    def test_plain_features_are_instance_scoped(self) -> None:
        registry_before = dict(vulkan_supported_ops)
        partitioner = build_webgpu_partitioner("8da4w+emb4")

        self.assertEqual(vulkan_supported_ops, registry_before)
        self.assertIn(
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            partitioner._inner.extra_op_features,
        )
        self.assertIn(
            exir_ops.edge.et_vk.gemma4_sdpa.default,
            partitioner._inner.extra_op_features,
        )
        # Assert against the GLOBAL registry, not a default partitioner's
        # instance map: that map is unconditionally empty, so the old form
        # passed even if the op were globally registered.
        self.assertNotIn(
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            vulkan_supported_ops,
        )
        self.assertNotIn(
            exir_ops.edge.et_vk.gemma4_sdpa.default,
            vulkan_supported_ops,
        )
        self.assertEqual(VulkanPartitioner().extra_op_features, {})

    def test_restricted_allowlist_includes_symbolic_select(self) -> None:
        allowlist = set(_webgpu_allowlist())
        self.assertIn(exir_ops.edge.et_vk.select_as_symint.default, allowlist)
        self.assertNotIn(exir_ops.edge.aten.mm.default, allowlist)
        self.assertNotIn(exir_ops.edge.aten.linear.default, allowlist)

    def test_emb8_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "emb8"):
            build_webgpu_partitioner("8da4w+emb8")


_QUERY_SHAPE = (1, 4, 8, 256)
_KV_SHAPE = (1, 4, 1, 256)
_MASK_SHAPE = (4, 4)
_ROPE_SHAPE = (1, 4, 8, 256)
_FREQS_SHAPE = (4, 128)


class _ScopedLayer(torch.nn.Module):
    pass


class _GraphProgram:
    """Stands in for the ExportedProgram surface the scoped rewriters read."""

    __slots__ = ("graph_module",)

    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        self.graph_module = graph_module


def _meta(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


def _placeholder(
    graph: torch.fx.Graph,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = _meta(shape, dtype)
    return node


# Each layout defeats a different stand-in for scope matching: `interleaved` a
# positional slice, `nested` an equality/prefix test on the module path.
_SCOPE_LAYOUTS = ("blocks", "interleaved", "nested")


def _scope_meta(scope: str, index: int, layout: str) -> dict[str, tuple[str, type]]:
    path = (
        f"decoder.{scope}_layers.{index}" if layout == "nested" else f"{scope}.{index}"
    )
    return {f"L__self___{scope}_{index}": (path, _ScopedLayer)}


def _scope_plan(target: int, assistant: int, unscoped: int, layout: str) -> list[str]:
    queues = [["target"] * target, ["assistant"] * assistant, [""] * unscoped]
    if layout != "interleaved":
        return [scope for queue in queues for scope in queue]
    plan: list[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                plan.append(queue.pop())
    return plan


def _module_paths(node: torch.fx.Node) -> list[str]:
    stack = node.meta.get("nn_module_stack") or {}
    return [entry[0] for entry in stack.values()]


def _nodes_with_target(graph: torch.fx.Graph, target: object) -> list[torch.fx.Node]:
    return [node for node in graph.nodes if node.target == target]


def _scoped_graph_module(
    graph: torch.fx.Graph,
    target_op: object,
    call_args: tuple[object, ...],
    result_shape: tuple[int, ...],
    plan: list[str],
    layout: str,
) -> torch.fx.GraphModule:
    emitted: dict[str, int] = {}
    calls: list[torch.fx.Node] = []
    for scope in plan:
        index = emitted.get(scope, 0)
        emitted[scope] = index + 1
        node = graph.call_function(target_op, call_args)
        node.meta["val"] = _meta(result_shape)
        if scope:
            node.meta["nn_module_stack"] = _scope_meta(scope, index, layout)
        calls.append(node)
    graph.output(tuple(calls))
    graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _sdpa_program(
    target: int, assistant: int, unscoped: int, layout: str = "blocks"
) -> _GraphProgram:
    graph = torch.fx.Graph()
    query = _placeholder(graph, "query", _QUERY_SHAPE)
    key = _placeholder(graph, "key", _KV_SHAPE)
    value = _placeholder(graph, "value", _KV_SHAPE)
    mask = _placeholder(graph, "mask", _MASK_SHAPE)
    return _GraphProgram(
        _scoped_graph_module(
            graph,
            exir_ops.edge.llama.custom_sdpa.default,
            (query, key, value, 0, mask, 0.0, False, 1.0),
            _QUERY_SHAPE,
            _scope_plan(target, assistant, unscoped, layout),
            layout,
        )
    )


def _rope_program(
    target: int, assistant: int, unscoped: int, layout: str = "blocks"
) -> _GraphProgram:
    graph = torch.fx.Graph()
    activations = _placeholder(graph, "x", _ROPE_SHAPE)
    freqs_cos = _placeholder(graph, "freqs_cos", _FREQS_SHAPE)
    freqs_sin = _placeholder(graph, "freqs_sin", _FREQS_SHAPE)
    return _GraphProgram(
        _scoped_graph_module(
            graph,
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            (activations, freqs_cos, freqs_sin, 0),
            _ROPE_SHAPE,
            _scope_plan(target, assistant, unscoped, layout),
            layout,
        )
    )


class _MtpPartitionerTest(unittest.TestCase):
    def setUp(self) -> None:
        # Deferred: the plain cases above must load without D8 or the custom-ops lib.
        import executorch.examples.models.gemma4.webgpu_partitioner as mtp
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

        self.mtp = mtp


class MTPScopedSDPATest(_MtpPartitionerTest):
    def test_exactly_the_target_scoped_sdpa_sites_are_rewritten(self) -> None:
        custom = exir_ops.edge.llama.custom_sdpa.default
        for layout in _SCOPE_LAYOUTS:
            with self.subTest(layout=layout):
                program = _sdpa_program(35, 8, 0, layout)
                planted = [
                    _module_paths(node)
                    for node in _nodes_with_target(program.graph_module.graph, custom)
                ]
                self.mtp._rewrite_mtp_sdpa(program)

                graph = program.graph_module.graph
                rewritten = _nodes_with_target(
                    graph, exir_ops.edge.et_vk.gemma4_sdpa.default
                )
                remaining = _nodes_with_target(graph, custom)
                self.assertEqual([_module_paths(node) for node in rewritten], planted)
                self.assertEqual(remaining, [])

    def test_sdpa_scope_counts_fail_closed(self) -> None:
        for target, assistant in ((34, 8), (36, 8), (35, 7), (35, 9)):
            with self.subTest(target=target, assistant=assistant):
                program = _sdpa_program(target, assistant, 0)
                with self.assertRaisesRegex(ValueError, "SDPA scope mismatch"):
                    self.mtp._rewrite_mtp_sdpa(program)
                self.assertEqual(
                    _nodes_with_target(
                        program.graph_module.graph,
                        exir_ops.edge.et_vk.gemma4_sdpa.default,
                    ),
                    [],
                )

    def test_unscoped_sdpa_is_rejected(self) -> None:
        program = _sdpa_program(35, 8, 1)
        with self.assertRaisesRegex(ValueError, "unscoped=1"):
            self.mtp._rewrite_mtp_sdpa(program)
        self.assertEqual(
            _nodes_with_target(
                program.graph_module.graph,
                exir_ops.edge.et_vk.gemma4_sdpa.default,
            ),
            [],
        )

    def test_sdpa_abi_and_argument_guards_reject_bad_calls(self) -> None:
        program = _sdpa_program(35, 8, 0)
        first = _nodes_with_target(
            program.graph_module.graph, exir_ops.edge.llama.custom_sdpa.default
        )[0]
        first.args = first.args[:7]
        with self.assertRaisesRegex(ValueError, "positional ABI"):
            self.mtp._rewrite_mtp_sdpa(program)

        program = _sdpa_program(35, 8, 0)
        first = _nodes_with_target(
            program.graph_module.graph, exir_ops.edge.llama.custom_sdpa.default
        )[0]
        first.args = (*first.args[:7], 0.125)
        with self.assertRaisesRegex(ValueError, "not WebGPU-compatible"):
            self.mtp._rewrite_mtp_sdpa(program)


class MTPScopedRoPETest(_MtpPartitionerTest):
    def test_the_official_rope_split_is_accepted_without_rewriting(self) -> None:
        rope = exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default
        for layout in _SCOPE_LAYOUTS:
            with self.subTest(layout=layout):
                program = _rope_program(20, 8, 0, layout)
                planted = [
                    _module_paths(node)
                    for node in _nodes_with_target(program.graph_module.graph, rope)
                ]

                self.mtp._replace_mtp_single_hf_rope(program)

                self.assertEqual(
                    [
                        _module_paths(node)
                        for node in _nodes_with_target(program.graph_module.graph, rope)
                    ],
                    planted,
                )

    def test_single_hf_rope_scope_counts_fail_closed(self) -> None:
        for target, assistant, unscoped in (
            (19, 8, 0),
            (21, 8, 0),
            (20, 7, 0),
            (20, 9, 0),
            (20, 8, 1),
        ):
            with self.subTest(target=target, assistant=assistant, unscoped=unscoped):
                program = _rope_program(target, assistant, unscoped)
                with self.assertRaisesRegex(
                    ValueError, "single-HF-RoPE scope mismatch"
                ):
                    self.mtp._replace_mtp_single_hf_rope(program)


def _topk_node(
    graph: torch.fx.Graph,
    *,
    source_shape: tuple[int, ...] = (1, 1, 2048),
    source_dtype: torch.dtype = torch.float32,
    values_shape: tuple[int, ...] = (1, 1, 32),
    indices_shape: tuple[int, ...] = (1, 1, 32),
    indices_dtype: torch.dtype = torch.int64,
    k: int = 32,
    dim: int = -1,
    largest: bool = True,
    sorted_values: bool = True,
    outputs: object | None = None,
) -> torch.fx.Node:
    source = _placeholder(graph, "scores", source_shape, source_dtype)
    node = graph.call_function(
        exir_ops.edge.aten.topk.default, (source, k, dim, largest, sorted_values)
    )
    node.meta["val"] = (
        outputs
        if outputs is not None
        else (_meta(values_shape), _meta(indices_shape, indices_dtype))
    )
    return node


def _scatter_node(
    graph: torch.fx.Graph,
    *,
    base_shape: tuple[int, ...] = (1, 1, 262144),
    base_dtype: torch.dtype = torch.float32,
    dim: int = -1,
    index_shape: tuple[int, ...] = (1, 1, 4096),
    index_dtype: torch.dtype = torch.int64,
    source_shape: tuple[int, ...] = (1, 1, 4096),
    source_dtype: torch.dtype = torch.float32,
    result_shape: tuple[int, ...] = (1, 1, 262144),
    result_dtype: torch.dtype = torch.float32,
    drop_source: bool = False,
) -> torch.fx.Node:
    base = _placeholder(graph, "base", base_shape, base_dtype)
    index = _placeholder(graph, "index", index_shape, index_dtype)
    source = _placeholder(graph, "source", source_shape, source_dtype)
    args = (base, dim, index) if drop_source else (base, dim, index, source)
    node = graph.call_function(exir_ops.edge.et_vk.scatter_src_unique.default, args)
    node.meta["val"] = _meta(result_shape, result_dtype)
    return node


class MTPExtraOpFeatureTest(_MtpPartitionerTest):
    def test_mtp_features_do_not_mutate_the_global_registry(self) -> None:
        registry_before = dict(vulkan_supported_ops)
        partitioner = self.mtp.build_webgpu_partitioner("8da4w+emb4", mode="mtp")

        self.assertEqual(vulkan_supported_ops, registry_before)
        self.assertEqual(set(vulkan_supported_ops), set(registry_before))
        self.assertIn(
            exir_ops.edge.aten.topk.default, partitioner._inner.extra_op_features
        )
        self.assertIn(
            exir_ops.edge.et_vk.scatter_src_unique.default,
            partitioner._inner.extra_op_features,
        )
        self.assertNotIn(
            exir_ops.edge.aten.topk.default, VulkanPartitioner().extra_op_features
        )
        self.assertNotIn(
            exir_ops.edge.et_vk.scatter_src_unique.default,
            VulkanPartitioner().extra_op_features,
        )

    def test_plain_mode_exposes_neither_scatter_nor_topk(self) -> None:
        plain_allowlist = set(_webgpu_allowlist())
        plain_features = _extra_op_features()
        mtp_allowlist = set(self.mtp._mtp_webgpu_allowlist())
        mtp_features = self.mtp.mtp_extra_op_features()

        for op in (
            exir_ops.edge.aten.topk.default,
            exir_ops.edge.et_vk.scatter_src_unique.default,
        ):
            self.assertNotIn(op, plain_allowlist)
            self.assertNotIn(op, plain_features)
            self.assertIn(op, mtp_allowlist)
            self.assertIn(op, mtp_features)
        self.assertTrue(plain_allowlist.issubset(mtp_allowlist))
        self.assertEqual(len(self.mtp._mtp_webgpu_allowlist()), len(mtp_allowlist))

    def test_uncertified_scatter_is_unreachable(self) -> None:
        self.assertNotIn(
            exir_ops.edge.et_vk.scatter_src_unique.default, vulkan_supported_ops
        )
        self.assertFalse(has_impl(exir_ops.edge.et_vk.scatter_src_unique.default))
        self.assertNotIn(exir_ops.edge.aten.scatter.src, vulkan_supported_ops)
        self.assertNotIn(
            exir_ops.edge.aten.scatter.src, self.mtp.mtp_extra_op_features()
        )
        self.assertNotIn(
            exir_ops.edge.aten.scatter.src, set(self.mtp._mtp_webgpu_allowlist())
        )
        self.assertNotIn(exir_ops.edge.aten.topk.default, vulkan_supported_ops)

    def test_topk_gate_accepts_only_the_official_qat_shape(self) -> None:
        gate = self.mtp.mtp_extra_op_features()[
            exir_ops.edge.aten.topk.default
        ].are_node_inputs_supported_fn
        graph = torch.fx.Graph()
        self.assertTrue(gate(_topk_node(graph)))

        for case, override in {
            "narrow_input": {"source_shape": (1, 1, 2047)},
            "wide_input": {"source_shape": (1, 1, 2049)},
            "rank_two_input": {"source_shape": (1, 2048)},
            "half_input": {"source_dtype": torch.float16},
            "short_values": {"values_shape": (1, 1, 31)},
            "long_values": {"values_shape": (1, 1, 33)},
            "short_indices": {"indices_shape": (1, 1, 31)},
            "float_indices": {"indices_dtype": torch.float32},
            "small_k": {"k": 31},
            "large_k": {"k": 33},
            "leading_dim": {"dim": 0},
            "non_negative_dim": {"dim": 2},
            "smallest": {"largest": False},
            "unsorted": {"sorted_values": False},
        }.items():
            with self.subTest(case=case):
                self.assertFalse(gate(_topk_node(graph, **override)))
        with self.subTest(case="single_output"):
            self.assertFalse(gate(_topk_node(graph, outputs=_meta((1, 1, 32)))))

    def test_scatter_gate_accepts_only_the_official_qat_shape(self) -> None:
        gate = self.mtp.mtp_extra_op_features()[
            exir_ops.edge.et_vk.scatter_src_unique.default
        ].are_node_inputs_supported_fn
        graph = torch.fx.Graph()
        self.assertTrue(gate(_scatter_node(graph)))

        for case, override in {
            "narrow_base": {"base_shape": (1, 1, 262143)},
            "wide_base": {"base_shape": (1, 1, 262145)},
            "half_base": {"base_dtype": torch.float16},
            "leading_dim": {"dim": 0},
            "non_negative_dim": {"dim": 2},
            "short_index": {"index_shape": (1, 1, 4095)},
            "long_index": {"index_shape": (1, 1, 4097)},
            "float_index": {"index_dtype": torch.float32},
            "short_source": {"source_shape": (1, 1, 4095)},
            "integer_source": {"source_dtype": torch.int64},
            "narrow_result": {"result_shape": (1, 1, 262143)},
            "integer_result": {"result_dtype": torch.int64},
            "missing_source": {"drop_source": True},
        }.items():
            with self.subTest(case=case):
                self.assertFalse(gate(_scatter_node(graph, **override)))

    def test_mtp_partitioner_rejects_unofficial_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "8da4w\\+emb4"):
            self.mtp.build_webgpu_partitioner("8da4w+emb8", mode="mtp")
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.mtp.build_webgpu_partitioner(
                "8da4w+emb4",
                mode="mtp",
                compile_options={"skip_bool_tensors": True},
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.mtp.build_webgpu_partitioner(
                "8da4w+emb4",
                mode="mtp",
                compile_options={"require_dynamic_shapes": False},
            )


def _certified_round(
    hidden: torch.Tensor,
    centroid_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    embedding_scales: torch.Tensor,
    ordering: torch.Tensor,
    output_template: torch.Tensor,
) -> torch.Tensor:
    scores = torch.nn.functional.linear(hidden, centroid_weight)
    selected = torch.topk(scores, 32, dim=-1, largest=True, sorted=True)[1]
    rows = torch.nn.functional.embedding(selected, ordering)
    converted = rows.to(torch.int64)
    index = converted.view(1, 1, 4096)
    flat_index = converted.view(4096)
    selected_embeddings = torch.ops.quantized_decomposed.embedding_4bit.dtype(
        embedding_weight,
        embedding_scales,
        None,
        -8,
        7,
        flat_index,
        dtype=torch.float32,
    )
    selected_transpose = selected_embeddings.view(1, 1, 4096, 256).transpose(2, 3)
    source = torch.matmul(hidden.unsqueeze(2), selected_transpose).squeeze(2)
    return output_template.scatter(-1, index, source)


class _CertifiedResidualFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("centroid_weight", torch.zeros(2048, 256))
        self.register_buffer(
            "embedding_weight", torch.zeros(262144, 128, dtype=torch.int8)
        )
        self.register_buffer("embedding_scales", torch.ones(262144, 1))
        self.register_buffer(
            "output_template",
            torch.full((1, 1, 262144), torch.finfo(torch.float32).min),
        )

    def _round(self, hidden: torch.Tensor, ordering: torch.Tensor) -> torch.Tensor:
        return _certified_round(
            hidden,
            self.centroid_weight,
            self.embedding_weight,
            self.embedding_scales,
            ordering,
            self.output_template,
        )


class _CertifiedResidualChain(_CertifiedResidualFixture):
    def __init__(self, ordering: torch.Tensor, persistent: bool) -> None:
        super().__init__()
        self.register_buffer(
            "token_ordering", ordering.to(torch.float32), persistent=persistent
        )

    def forward(
        self,
        first_hidden: torch.Tensor,
        second_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._round(first_hidden, self.token_ordering),
            self._round(second_hidden, self.token_ordering),
        )


class _NonconstantOrderingChain(_CertifiedResidualFixture):
    def forward(
        self,
        first_hidden: torch.Tensor,
        second_hidden: torch.Tensor,
        ordering: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._round(first_hidden, ordering),
            self._round(second_hidden, ordering),
        )


def _identity_ordering() -> torch.Tensor:
    return torch.arange(262144, dtype=torch.int64).reshape(2048, 128)


def _chain_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, 1, 256),
        torch.ones(1, 1, 256),
    )


def _export_certified_chain(
    persistent: bool = False,
) -> torch.export.ExportedProgram:
    return torch.export.export(
        _CertifiedResidualChain(_identity_ordering(), persistent),
        _chain_inputs(),
        strict=True,
    ).run_decompositions({})


def _export_nonconstant_ordering_chain() -> torch.export.ExportedProgram:
    return torch.export.export(
        _NonconstantOrderingChain(),
        (*_chain_inputs(), _identity_ordering().to(torch.float32)),
        strict=True,
    ).run_decompositions({})


def _mtp_delegation_tags(
    program: torch.export.ExportedProgram,
    partitioner: Gemma4WebGPUPartitioner,
) -> dict[object, list[object]]:
    # The production-shaped matmul decomposes through non-resizable
    # expand_copy nodes. This certifier unit owns op eligibility; real-model
    # serializer closure is covered by the export and artifact gates.
    edge = to_edge(
        program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    result = partitioner.partition(edge)
    targets = (
        exir_ops.edge.aten.topk.default,
        exir_ops.edge.et_vk.scatter_src_unique.default,
    )
    return {
        target: [
            node.meta.get("delegation_tag")
            for node in result.tagged_exported_program.graph.nodes
            if node.target == target
        ]
        for target in targets
    }


class MTPCertifiedScatterTest(_MtpPartitionerTest):
    def test_exactly_two_certified_sites_are_rewritten(self) -> None:
        for persistent in (False, True):
            with self.subTest(persistent_ordering_buffer=persistent):
                program = _export_certified_chain(persistent)
                rewrites = self.mtp.rewrite_certified_unique_scatter(
                    program, _identity_ordering(), expected_chains=2
                )

                self.assertEqual(rewrites, 2)
                self.assertEqual(
                    len(
                        _nodes_with_target(
                            program.graph,
                            torch.ops.et_vk.scatter_src_unique.default,
                        )
                    ),
                    2,
                )
                self.assertEqual(
                    _nodes_with_target(program.graph, torch.ops.aten.scatter.src),
                    [],
                )

    def test_a_rejected_rewrite_leaves_the_graph_untouched(self) -> None:
        duplicated = _identity_ordering().reshape(-1).clone()
        duplicated[0] = duplicated[1]
        rejections = {
            "chain_count_one": (_identity_ordering(), 1, "residual topology mismatch"),
            "chain_count_three": (
                _identity_ordering(),
                3,
                "residual topology mismatch",
            ),
            "chain_count_zero": (
                _identity_ordering(),
                0,
                "residual topology mismatch",
            ),
            "foreign_ordering": (
                _identity_ordering().reshape(-1).roll(1).reshape(2048, 128),
                2,
                "token-ordering identity mismatch",
            ),
            "non_permutation_ordering": (
                duplicated.reshape(2048, 128),
                2,
                "exact permutation",
            ),
        }
        for case, (ordering, expected_chains, message) in rejections.items():
            with self.subTest(case=case):
                program = _export_certified_chain()
                with self.assertRaisesRegex(ValueError, message):
                    self.mtp.rewrite_certified_unique_scatter(
                        program, ordering, expected_chains=expected_chains
                    )
                self.assertEqual(
                    len(_nodes_with_target(program.graph, torch.ops.aten.scatter.src)),
                    2,
                )

    def test_a_nonconstant_ordering_source_is_rejected(self) -> None:
        program = _export_nonconstant_ordering_chain()
        with self.assertRaisesRegex(ValueError, "token-ordering identity mismatch"):
            self.mtp.rewrite_certified_unique_scatter(
                program, _identity_ordering(), expected_chains=2
            )
        self.assertEqual(
            len(_nodes_with_target(program.graph, torch.ops.aten.scatter.src)), 2
        )

    def test_mtp_partitioner_tags_certified_topk_and_scatter(self) -> None:
        program = _export_certified_chain()
        self.mtp.rewrite_certified_unique_scatter(
            program, _identity_ordering(), expected_chains=2
        )
        tags = _mtp_delegation_tags(
            program,
            self.mtp.build_webgpu_partitioner("8da4w+emb4", mode="mtp"),
        )

        for op in (
            exir_ops.edge.aten.topk.default,
            exir_ops.edge.et_vk.scatter_src_unique.default,
        ):
            self.assertEqual(len(tags[op]), 2)
            self.assertTrue(all(tag is not None for tag in tags[op]))

    def test_dropping_an_mtp_feature_breaks_the_lowering_gate(self) -> None:
        for op in (
            exir_ops.edge.aten.topk.default,
            exir_ops.edge.et_vk.scatter_src_unique.default,
        ):
            with self.subTest(op=op.__name__):
                program = _export_certified_chain()
                self.mtp.rewrite_certified_unique_scatter(
                    program, _identity_ordering(), expected_chains=2
                )
                partitioner = self.mtp.build_webgpu_partitioner(
                    "8da4w+emb4", mode="mtp"
                )
                del partitioner._inner.extra_op_features[op]
                tags = _mtp_delegation_tags(program, partitioner)
                self.assertEqual(tags[op], [None, None])
                other = (
                    exir_ops.edge.et_vk.scatter_src_unique.default
                    if op == exir_ops.edge.aten.topk.default
                    else exir_ops.edge.aten.topk.default
                )
                self.assertEqual(len(tags[other]), 2)
                self.assertTrue(all(tag is not None for tag in tags[other]))
