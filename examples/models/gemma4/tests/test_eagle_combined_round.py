# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import contextlib
import dataclasses
import unittest
from collections.abc import Iterator
from pathlib import Path

import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

import torch

from executorch.examples.models.gemma4.eagle_webgpu_round import (
    _expected_mutation_contract,
    _rewrite_negative_select_as_symint,
    export_k2_round_program,
    Gemma4K2Target,
    K2GPUResidentRound,
    K2LongestPrefixSelector,
    validate_k2_round_abi,
)
from executorch.examples.models.gemma4.export_speculative import (
    _lower_k2_round,
    build_k2_round_program,
)
from executorch.examples.models.gemma4.webgpu_partitioner import (
    _is_official_qat_topk,
    _is_official_qat_unique_scatter,
    build_webgpu_partitioner,
    mtp_extra_op_features,
    rewrite_certified_unique_scatter,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import (
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)


_ABI_MAX_INPUT_LEN = 8
_ABI_MAX_DONOR_LEN = 8960
_ABI_MUTATION_COUNT = 31
_ABI_SDPA_COUNT = 43
_ABI_ARGMAX_COUNT = 3
_ABI_TOPK_COUNT = 2
_ABI_SCATTER_COUNT = 2

_FULL_DONOR_LAYER = 14
_SLIDING_DONOR_LAYER = 13
_DONOR_LAYER_COUNT = 16

_DONOR_LENGTHS = (2, 511, 512, 513, 514, 8960)


def _mutation_buffer_names(
    count: int = _ABI_MUTATION_COUNT,
    seed_names: tuple[str, ...] = ("seed_feature",),
) -> tuple[str, ...]:
    filler = tuple(f"donor_cache_{index}" for index in range(count - len(seed_names)))
    return seed_names + filler


class _K2AbiFixture(torch.nn.Module):
    """Smallest module whose exported graph reproduces the `k2_round` ABI."""

    def __init__(
        self,
        *,
        max_input_len: int = _ABI_MAX_INPUT_LEN,
        max_donor_len: int = _ABI_MAX_DONOR_LEN,
        mutation_names: tuple[str, ...] | None = None,
        sdpa_count: int = _ABI_SDPA_COUNT,
        argmax_count: int = _ABI_ARGMAX_COUNT,
        topk_count: int = _ABI_TOPK_COUNT,
        scatter_count: int = _ABI_SCATTER_COUNT,
    ) -> None:
        super().__init__()
        self.max_donor_len = max_donor_len
        self.sdpa_count = sdpa_count
        self.argmax_count = argmax_count
        self.topk_count = topk_count
        self.scatter_count = scatter_count
        self.mutation_names: tuple[str, ...] = (
            _mutation_buffer_names() if mutation_names is None else mutation_names
        )
        self.register_buffer(
            "round_tail", torch.zeros((1, max_input_len - 3)), persistent=False
        )
        self.register_buffer(
            "donor_pool", torch.zeros((1, max_donor_len, 1, 1)), persistent=False
        )
        self.register_buffer(
            "attn_mask", torch.zeros((1, max_donor_len)), persistent=False
        )
        self.register_buffer("query", torch.zeros((1, 1, 1, 1)), persistent=False)
        self.register_buffer("logits", torch.zeros((1, 1, 8)), persistent=False)
        self.register_buffer("scatter_row", torch.zeros((1, 4)), persistent=False)
        self.register_buffer(
            "scatter_index", torch.zeros((1, 2), dtype=torch.int64), persistent=False
        )
        for name in self.mutation_names:
            shape = (1, 1, 1, 1536) if name == "seed_feature" else (1, 1, 1, 1)
            self.register_buffer(name, torch.zeros(shape))

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        is_round: torch.Tensor,
        donor_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        donor_k = donor_length[0, 0].item()
        torch._check_is_size(donor_k)
        torch._check(donor_k >= 2)
        torch._check(donor_k <= self.max_donor_len)
        donor = self.donor_pool.narrow(1, 0, donor_k)
        mask = self.attn_mask.narrow(1, 0, donor_k)
        live = (
            input_ids.to(torch.float32).sum()
            + input_pos.to(torch.float32).sum()
            + is_round.to(torch.float32).sum()
        )
        attention = self.query + live
        for _ in range(self.sdpa_count):
            attention = torch.ops.llama.custom_sdpa.default(
                attention, donor, donor, 0, mask, 0.0, False, 1.0
            )
        scores = self.logits + attention.reshape(1, 1, 1)
        for _ in range(self.topk_count):
            scores = scores + torch.topk(scores, 2, dim=-1).values.sum()
        scattered = self.scatter_row
        for _ in range(self.scatter_count):
            scattered = scattered.scatter(
                -1, self.scatter_index, scores.reshape(1, -1)[:, :2]
            )
        greedy: list[torch.Tensor] = []
        probe = scores
        for _ in range(self.argmax_count):
            greedy.append(torch.argmax(probe, dim=-1))
            probe = probe + 1.0
        next_feature = (live + attention.reshape(1, 1)).reshape(1, 1, 1).expand(
            1, 1, 1536
        )
        torch.ops.llama.update_cache.default(
            next_feature.unsqueeze(2), self.seed_feature, 0
        )
        cache_value = live.reshape(1, 1, 1, 1)
        for name in self.mutation_names[1:]:
            torch.ops.llama.update_cache.default(cache_value, getattr(self, name), 0)
        head = greedy[0]
        bonus = head + 1
        for value in greedy[1:]:
            bonus = bonus + value
        seed = self.get_buffer("seed_feature")
        return (
            torch.cat((head, head), dim=1),
            torch.cat((head, head, head), dim=1),
            head.reshape(1),
            bonus,
            (seed[..., 0] + scattered.sum() + donor.sum()).reshape(1, 1),
        )


def _export_abi_fixture(**overrides: int) -> torch.export.ExportedProgram:
    fixture = _K2AbiFixture(**overrides).eval()
    program = export_k2_round_program(
        # pyre-ignore[6]: the ABI fixture duck-types `K2GPUResidentRound`.
        fixture,
        _ABI_MAX_INPUT_LEN,
    )
    expected = _expected_mutation_contract(_ABI_MAX_DONOR_LEN)
    signature = program.graph_signature
    nodes = {node.name: node for node in program.graph.nodes}
    placeholders = {target: name for name, target in signature.inputs_to_buffers.items()}
    mutation_specs: list[OutputSpec] = []
    for source_target, record in zip(fixture.mutation_names, expected):
        name = placeholders[source_target]
        nodes[name].meta["val"] = torch.empty(
            tuple(record["shape"]), dtype=torch.float32, device="meta"
        )
        mutation_specs.append(
            OutputSpec(
                kind=OutputKind.BUFFER_MUTATION,
                arg=TensorArgument(name=name),
                target=record["logicalTarget"],
            )
        )
    signature.output_specs[:] = mutation_specs + [
        spec for spec in signature.output_specs if spec.kind == OutputKind.USER_OUTPUT
    ]
    return program


def _range_bounds(program: torch.export.ExportedProgram) -> set[tuple[int, int]]:
    bounds: set[tuple[int, int]] = set()
    for value in program.range_constraints.values():
        try:
            bounds.add((int(value.lower), int(value.upper)))
        except (OverflowError, TypeError, ValueError):
            continue
    return bounds


@contextlib.contextmanager
def _reversible_abi_edit(program: torch.export.ExportedProgram) -> Iterator[None]:
    """Undo signature / range / node-meta edits so one export can serve many mutants."""
    signature = program.graph_signature
    saved_inputs = list(signature.input_specs)
    saved_outputs = list(signature.output_specs)
    saved_ranges = dict(program.range_constraints)
    saved_vals = {node.name: node.meta.get("val") for node in program.graph.nodes}
    try:
        yield
    finally:
        signature.input_specs[:] = saved_inputs
        signature.output_specs[:] = saved_outputs
        program.range_constraints.clear()
        program.range_constraints.update(saved_ranges)
        for node in program.graph.nodes:
            if node.name in saved_vals:
                node.meta["val"] = saved_vals[node.name]


class K2RoundAbiTest(unittest.TestCase):
    """`validate_k2_round_abi` pins the exact `k2_round` graph ABI."""

    program: torch.export.ExportedProgram

    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _export_abi_fixture()

    def _validate(self, program: torch.export.ExportedProgram) -> object:
        return validate_k2_round_abi(
            program,
            max_input_len=_ABI_MAX_INPUT_LEN,
            max_donor_len=_ABI_MAX_DONOR_LEN,
        )

    def _nodes(self) -> dict[str, torch.fx.Node]:
        return {node.name: node for node in self.program.graph.nodes}

    def _mutation_indices(self) -> list[int]:
        return [
            index
            for index, spec in enumerate(self.program.graph_signature.output_specs)
            if spec.kind == OutputKind.BUFFER_MUTATION
        ]

    def test_abi_evidence_reports_the_documented_census(self) -> None:
        evidence = self._validate(self.program)
        self.assertEqual(
            set(evidence),
            {
                "bufferMutationCount",
                "donorViewOrder",
                "inputOrder",
                "mutationOrder",
                "operatorCounts",
                "outputOrder",
                "seedMutationCount",
                "stateAlias",
            },
        )
        self.assertEqual(evidence["bufferMutationCount"], 31)
        self.assertEqual(
            evidence["operatorCounts"],
            {
                "aten.argmax.default": 3,
                "aten.scatter.src": 2,
                "aten.topk.default": 2,
                "llama.custom_sdpa.default": 43,
                "llama.update_cache.default": 31,
            },
        )
        self.assertEqual(evidence["seedMutationCount"], 1)

    def test_user_inputs_are_ordered_ids_pos_round_donor(self) -> None:
        self.assertEqual(
            tuple(self.program.graph_signature.user_inputs),
            ("input_ids", "input_pos", "is_round", "donor_length"),
        )

    def test_user_input_shapes_and_dtypes_are_exact(self) -> None:
        nodes = self._nodes()
        for name in ("input_ids", "input_pos", "is_round", "donor_length"):
            self.assertEqual(nodes[name].op, "placeholder", name)
            self.assertEqual(nodes[name].meta["val"].dtype, torch.int64, name)
        self.assertEqual(len(nodes["input_ids"].meta["val"].shape), 2)
        self.assertEqual(nodes["input_ids"].meta["val"].shape[0], 1)
        self.assertEqual(len(nodes["input_pos"].meta["val"].shape), 1)
        self.assertEqual(tuple(nodes["is_round"].meta["val"].shape), (1,))
        self.assertEqual(tuple(nodes["donor_length"].meta["val"].shape), (1, 1))

    def test_input_order_permutation_is_rejected(self) -> None:
        specs = self.program.graph_signature.input_specs
        first, second = (
            index
            for index, spec in enumerate(specs)
            if spec.kind == InputKind.USER_INPUT
            and spec.arg.name in ("input_ids", "input_pos")
        )
        with _reversible_abi_edit(self.program):
            specs[first], specs[second] = specs[second], specs[first]
            with self.assertRaisesRegex(ValueError, "user-input order mismatch"):
                self._validate(self.program)

    def test_input_dtype_and_shape_regressions_are_rejected(self) -> None:
        mutants = {
            "is_round": torch.zeros((1,), dtype=torch.int32),
            "donor_length": torch.zeros((1,), dtype=torch.int64),
            "input_ids": torch.zeros((2, 3), dtype=torch.int64),
        }
        nodes = self._nodes()
        for name, mutant in mutants.items():
            with self.subTest(name=name), _reversible_abi_edit(self.program):
                nodes[name].meta["val"] = mutant
                with self.assertRaisesRegex(ValueError, f"K=2 {name}"):
                    self._validate(self.program)

    def test_sequence_dimension_is_one_shared_symbol(self) -> None:
        nodes = self._nodes()
        input_ids = nodes["input_ids"].meta["val"]
        input_pos = nodes["input_pos"].meta["val"]
        self.assertEqual(str(input_ids.shape[1]), str(input_pos.shape[0]))
        self.assertNotEqual(str(input_ids.shape[1]), str(input_ids.shape[0]))
        with _reversible_abi_edit(self.program):
            nodes["input_pos"].meta["val"] = torch.zeros((3,), dtype=torch.int64)
            with self.assertRaisesRegex(ValueError, "dynamic dimensions differ"):
                self._validate(self.program)

    def test_user_outputs_are_ordered_and_typed(self) -> None:
        nodes = self._nodes()
        outputs = list(self.program.graph_signature.user_outputs)
        self.assertEqual(len(outputs), 5)
        expected = (
            ((1, 2), torch.int64),
            ((1, 3), torch.int64),
            ((1,), torch.int64),
            ((1, 1), torch.int64),
            ((1, 1), torch.float32),
        )
        for name, (shape, dtype) in zip(outputs, expected):
            value = nodes[str(name)].meta["val"]
            self.assertEqual(tuple(value.shape), shape, name)
            self.assertEqual(value.dtype, dtype, name)

    def test_output_order_permutation_is_rejected(self) -> None:
        specs = self.program.graph_signature.output_specs
        user = [
            index
            for index, spec in enumerate(specs)
            if spec.kind == OutputKind.USER_OUTPUT
        ]
        with _reversible_abi_edit(self.program):
            first, last = user[0], user[-1]
            specs[first], specs[last] = specs[last], specs[first]
            with self.assertRaisesRegex(ValueError, "K=2 candidates"):
                self._validate(self.program)

    def test_state_probe_must_stay_float32(self) -> None:
        probe = str(list(self.program.graph_signature.user_outputs)[-1])
        nodes = self._nodes()
        with _reversible_abi_edit(self.program):
            nodes[probe].meta["val"] = torch.zeros((1, 1), dtype=torch.int64)
            with self.assertRaisesRegex(ValueError, "K=2 state_probe dtype"):
                self._validate(self.program)

    def test_missing_user_output_is_rejected(self) -> None:
        specs = self.program.graph_signature.output_specs
        with _reversible_abi_edit(self.program):
            dropped = next(
                index
                for index, spec in enumerate(specs)
                if spec.kind == OutputKind.USER_OUTPUT
            )
            del specs[dropped]
            with self.assertRaisesRegex(ValueError, "user-output count mismatch"):
                self._validate(self.program)

    def test_mutation_census_is_thirty_one_with_one_seed(self) -> None:
        specs = self.program.graph_signature.output_specs
        mutations = self._mutation_indices()
        targets = [str(specs[index].target) for index in mutations]
        self.assertEqual(len(mutations), 31)
        self.assertEqual(
            [target for target in targets if target.endswith("seed_feature")],
            ["seed_feature"],
        )
        self.assertEqual(len(set(targets)), 31)

    def test_dropped_mutation_is_rejected(self) -> None:
        specs = self.program.graph_signature.output_specs
        with _reversible_abi_edit(self.program):
            del specs[self._mutation_indices()[-1]]
            with self.assertRaisesRegex(ValueError, "output-spec order mismatch"):
                self._validate(self.program)

    def test_second_seed_feature_mutation_is_rejected(self) -> None:
        specs = self.program.graph_signature.output_specs
        with _reversible_abi_edit(self.program):
            victim = next(
                index
                for index in self._mutation_indices()
                if not str(specs[index].target).endswith("seed_feature")
            )
            specs[victim] = dataclasses.replace(
                specs[victim], target="assistant.seed_feature"
            )
            with self.assertRaisesRegex(ValueError, "mutation target order mismatch"):
                self._validate(self.program)

    def test_duplicate_mutation_target_is_rejected(self) -> None:
        specs = self.program.graph_signature.output_specs
        with _reversible_abi_edit(self.program):
            non_seed = [
                index
                for index in self._mutation_indices()
                if not str(specs[index].target).endswith("seed_feature")
            ]
            specs[non_seed[1]] = dataclasses.replace(
                specs[non_seed[1]], target=str(specs[non_seed[0]].target)
            )
            with self.assertRaisesRegex(ValueError, "mutation target order mismatch"):
                self._validate(self.program)

    def test_operator_census_is_exact(self) -> None:
        mutants = {
            "llama.custom_sdpa.default": {"sdpa_count": _ABI_SDPA_COUNT - 1},
            "aten.argmax.default": {"argmax_count": _ABI_ARGMAX_COUNT - 1},
            "aten.topk.default": {"topk_count": _ABI_TOPK_COUNT - 1},
            "aten.scatter.src": {"scatter_count": _ABI_SCATTER_COUNT - 1},
        }
        for target, override in mutants.items():
            with self.subTest(target=target):
                program = _export_abi_fixture(**override)
                with self.assertRaisesRegex(ValueError, f"{target} count mismatch"):
                    self._validate(program)

    def test_both_dynamic_ranges_must_be_present(self) -> None:
        bounds = _range_bounds(self.program)
        self.assertIn((1, _ABI_MAX_INPUT_LEN), bounds)
        self.assertIn((2, _ABI_MAX_DONOR_LEN), bounds)
        for missing in ((1, _ABI_MAX_INPUT_LEN), (2, _ABI_MAX_DONOR_LEN)):
            with self.subTest(missing=missing), _reversible_abi_edit(self.program):
                constraints = self.program.range_constraints
                for symbol, value in list(constraints.items()):
                    try:
                        current = (int(value.lower), int(value.upper))
                    except (OverflowError, TypeError, ValueError):
                        continue
                    if current == missing:
                        del constraints[symbol]
                with self.assertRaisesRegex(ValueError, "missing dynamic range"):
                    self._validate(self.program)


class _DonorKVCache(torch.nn.Module):
    def __init__(self, offset: float, max_seq_len: int) -> None:
        super().__init__()
        base = torch.arange(max_seq_len * 8, dtype=torch.float32).reshape(
            1, max_seq_len, 2, 4
        )
        self.register_buffer("k_cache", base + offset)
        self.register_buffer("v_cache", base + offset + 0.5)


class _DonorAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        is_donor: bool = False,
        is_sliding: bool = False,
        kv_cache: _DonorKVCache | None = None,
    ) -> None:
        super().__init__()
        self.is_kv_donor_layer = is_donor
        self.is_sliding = is_sliding
        self.kv_cache = kv_cache


class _DonorLayer(torch.nn.Module):
    def __init__(self, self_attn: _DonorAttention) -> None:
        super().__init__()
        self.self_attn = self_attn


class _DonorTextModel(torch.nn.Module):
    def __init__(
        self,
        *,
        full_index: int = _FULL_DONOR_LAYER,
        sliding_index: int = _SLIDING_DONOR_LAYER,
        max_seq_len: int = 32,
        drop_full_cache: bool = False,
    ) -> None:
        super().__init__()
        layers: list[_DonorLayer] = []
        for index in range(_DONOR_LAYER_COUNT):
            if index == full_index:
                layers.append(
                    _DonorLayer(
                        _DonorAttention(
                            is_donor=True,
                            is_sliding=False,
                            kv_cache=(
                                None
                                if drop_full_cache
                                else _DonorKVCache(0.0, max_seq_len)
                            ),
                        )
                    )
                )
            elif index == sliding_index:
                layers.append(
                    _DonorLayer(
                        _DonorAttention(
                            is_donor=True,
                            is_sliding=True,
                            kv_cache=_DonorKVCache(1000.0, max_seq_len),
                        )
                    )
                )
            else:
                layers.append(_DonorLayer(_DonorAttention()))
        self.self_decoder = torch.nn.Module()
        self.self_decoder.layers = torch.nn.ModuleList(layers)


class DonorTopologyTest(unittest.TestCase):
    """`Gemma4K2Target` binds exactly one full donor (14) and one sliding donor (13)."""

    def test_official_topology_binds_layers_thirteen_and_fourteen(self) -> None:
        target = Gemma4K2Target(_DonorTextModel())
        self.assertEqual(target.full_donor_index, 14)
        self.assertEqual(target.sliding_donor_index, 13)

    def test_shifted_donor_layers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "donor layer mismatch"):
            Gemma4K2Target(_DonorTextModel(full_index=15))
        with self.assertRaisesRegex(ValueError, "donor layer mismatch"):
            Gemma4K2Target(_DonorTextModel(sliding_index=12))

    def test_duplicate_or_absent_donors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "donor topology mismatch"):
            Gemma4K2Target(_DonorTextModel(sliding_index=_FULL_DONOR_LAYER))
        empty = _DonorTextModel()
        for layer in empty.self_decoder.layers:
            layer.self_attn.is_kv_donor_layer = False
        with self.assertRaisesRegex(ValueError, "donor topology mismatch"):
            Gemma4K2Target(empty)

    def test_donor_without_kv_cache_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "donor has no KV cache"):
            Gemma4K2Target(_DonorTextModel(drop_full_cache=True))


class DonorViewTest(unittest.TestCase):
    """Donor views are `cache[:, :length]` in BHKD order fk / fv / sk / sv."""

    def test_views_are_prefix_slices_in_bhkd_layout(self) -> None:
        text_model = _DonorTextModel(max_seq_len=32)
        target = Gemma4K2Target(text_model)
        layers = text_model.self_decoder.layers
        full = layers[_FULL_DONOR_LAYER].self_attn.kv_cache
        sliding = layers[_SLIDING_DONOR_LAYER].self_attn.kv_cache
        for length in (2, 17, 32):
            with self.subTest(length=length):
                views = target.donor_views(torch.tensor([[length]], dtype=torch.int64))
                expected = (
                    full.k_cache[:, :length].permute(0, 2, 1, 3),
                    full.v_cache[:, :length].permute(0, 2, 1, 3),
                    sliding.k_cache[:, :length].permute(0, 2, 1, 3),
                    sliding.v_cache[:, :length].permute(0, 2, 1, 3),
                )
                self.assertEqual(len(views), 4)
                for index, (view, reference) in enumerate(zip(views, expected)):
                    self.assertEqual(tuple(view.shape), (1, 2, length, 4), index)
                    self.assertTrue(torch.equal(view, reference), index)

    def test_view_order_is_not_interchangeable(self) -> None:
        target = Gemma4K2Target(_DonorTextModel(max_seq_len=32))
        full_k, full_v, sliding_k, sliding_v = target.donor_views(
            torch.tensor([[8]], dtype=torch.int64)
        )
        for left, right in (
            (full_k, full_v),
            (full_k, sliding_k),
            (sliding_k, sliding_v),
        ):
            self.assertFalse(torch.equal(left, right))

    def test_donor_length_below_two_is_rejected(self) -> None:
        target = Gemma4K2Target(_DonorTextModel(max_seq_len=32))
        with self.assertRaises(RuntimeError):
            target.donor_views(torch.tensor([[1]], dtype=torch.int64))


class _RecordingTarget(torch.nn.Module):
    """Stand-in target that reports the donor length it was handed."""

    def __init__(self, *, donor_shrink: int = 0, head_dim: int = 2) -> None:
        super().__init__()
        self.donor_shrink = donor_shrink
        self.head_dim = head_dim
        self.donor_lengths: list[int] = []

    def donor_views(
        self, donor_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(donor_length[0, 0].item())
        self.donor_lengths.append(length)
        width = length - self.donor_shrink
        return (
            torch.full((1, 1, width, self.head_dim), 0.0),
            torch.full((1, 1, width, self.head_dim), 1.0),
            torch.full((1, 1, width, self.head_dim), 2.0),
            torch.full((1, 1, width, self.head_dim), 3.0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        is_round: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, input_pos, is_round
        return (
            torch.tensor([[21, 22, 23]], dtype=torch.long),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        )


class _ZeroEmbedding(torch.nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*tokens.shape, 2), dtype=torch.float32)


class _TokenEmbedding(torch.nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 2)


class _ScaleDroppingEmbedding(_TokenEmbedding):
    """Mutant: the caller-applied scale is cancelled on the second draft step."""

    def __init__(self, embed_scale: float) -> None:
        super().__init__()
        self.embed_scale = embed_scale
        self.calls = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = super().forward(tokens)
        self.calls += 1
        return embedding if self.calls == 1 else embedding / self.embed_scale


class _RecordingAssistant(torch.nn.Module):
    def __init__(self, first_token: int = 21) -> None:
        super().__init__()
        self.first_token = first_token
        self.inputs: list[torch.Tensor] = []
        self.positions: list[list[list[int]]] = []
        self.donor_shapes: list[tuple[tuple[int, ...], ...]] = []
        self.donor_markers: list[tuple[float, ...]] = []

    def forward(
        self,
        inputs: torch.Tensor,
        position_ids: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        donors = (full_k, full_v, sliding_k, sliding_v)
        self.inputs.append(inputs.clone())
        self.positions.append(position_ids.tolist())
        self.donor_shapes.append(tuple(tuple(donor.shape) for donor in donors))
        self.donor_markers.append(
            tuple(float(donor.reshape(-1)[0].item()) for donor in donors)
        )
        logits = torch.zeros((1, 1, 32), dtype=torch.float32)
        logits[..., self.first_token + len(self.inputs) - 1] = 1.0
        return logits, inputs[..., :2]


class _EagerK2Round(K2GPUResidentRound):
    """Keeps eager rounds off `llama.update_cache` so its dep is exercised once."""

    def update_seed_feature(self, next_feature: torch.Tensor) -> torch.Tensor:
        return next_feature.unsqueeze(2)


def _build_round(
    target: torch.nn.Module,
    embedding: torch.nn.Module,
    assistant: torch.nn.Module,
    *,
    embed_scale: float = 1.0,
    round_class: type[K2GPUResidentRound] = _EagerK2Round,
) -> K2GPUResidentRound:
    return round_class(
        target,
        embedding,
        assistant,
        hidden_size=2,
        max_input_len=3,
        max_donor_len=_ABI_MAX_DONOR_LEN,
        embed_scale=embed_scale,
    )


def _run_round(
    module: K2GPUResidentRound, donor_length: int, *, is_round: int = 1
) -> tuple[torch.Tensor, ...]:
    return module(
        torch.tensor([[10, 0, 0]], dtype=torch.long),
        torch.arange(donor_length, donor_length + 3, dtype=torch.long),
        torch.tensor([is_round], dtype=torch.long),
        torch.tensor([[donor_length]], dtype=torch.long),
    )


class K2RoundStepTest(unittest.TestCase):
    """K=2 assistant steps advance from `donor_length - 1` to `donor_length`."""

    def _assert_advancing_positions_and_full_donors(
        self, assistant: _RecordingAssistant, donor_length: int, head_dim: int = 2
    ) -> None:
        self.assertEqual(len(assistant.positions), 2)
        self.assertEqual(
            assistant.positions, [[[donor_length - 1]], [[donor_length]]]
        )
        expected_shapes = tuple((1, 1, donor_length, head_dim) for _ in range(4))
        self.assertEqual(assistant.donor_shapes, [expected_shapes, expected_shapes])
        self.assertEqual(
            assistant.donor_markers, [(0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0, 3.0)]
        )

    def test_positions_advance_and_donors_keep_full_length(self) -> None:
        for donor_length in _DONOR_LENGTHS:
            with self.subTest(donor_length=donor_length):
                target = _RecordingTarget()
                assistant = _RecordingAssistant()
                module = _build_round(target, _ZeroEmbedding(), assistant)
                _run_round(module, donor_length)
                self.assertEqual(target.donor_lengths, [donor_length])
                self._assert_advancing_positions_and_full_donors(assistant, donor_length)

    def test_donor_shrink_mutant_is_rejected(self) -> None:
        for donor_length in (2, 512, 8960):
            with self.subTest(donor_length=donor_length):
                assistant = _RecordingAssistant()
                module = _build_round(
                    _RecordingTarget(donor_shrink=1), _ZeroEmbedding(), assistant
                )
                _run_round(module, donor_length)
                with self.assertRaises(AssertionError):
                    self._assert_advancing_positions_and_full_donors(
                        assistant, donor_length
                    )

    def test_position_oracle_rejects_a_shifted_donor_length(self) -> None:
        donor_length = 512
        assistant = _RecordingAssistant()
        module = _build_round(_RecordingTarget(), _ZeroEmbedding(), assistant)
        _run_round(module, donor_length)
        for shifted in (donor_length - 1, donor_length + 1):
            with self.subTest(shifted=shifted):
                with self.assertRaises(AssertionError):
                    self._assert_advancing_positions_and_full_donors(
                        assistant, shifted
                    )


class K2EmbeddingScaleTest(unittest.TestCase):
    """`embed_scale` multiplies both draft embeddings and must be finite / positive."""

    def _assert_both_drafts_scaled(
        self, assistant: _RecordingAssistant, embed_scale: float
    ) -> None:
        self.assertEqual(len(assistant.inputs), 2)
        self.assertTrue(
            torch.equal(
                assistant.inputs[0][..., :2], torch.full((1, 1, 2), 10.0 * embed_scale)
            )
        )
        self.assertTrue(
            torch.equal(
                assistant.inputs[1][..., :2], torch.full((1, 1, 2), 21.0 * embed_scale)
            )
        )

    def test_scale_is_applied_to_both_draft_embeddings(self) -> None:
        for embed_scale in (0.5, 2.0):
            with self.subTest(embed_scale=embed_scale):
                assistant = _RecordingAssistant()
                module = _build_round(
                    _RecordingTarget(),
                    _TokenEmbedding(),
                    assistant,
                    embed_scale=embed_scale,
                )
                _run_round(module, 2)
                self._assert_both_drafts_scaled(assistant, embed_scale)

    def test_dropping_the_scale_on_the_second_draft_is_rejected(self) -> None:
        embed_scale = 0.5
        assistant = _RecordingAssistant()
        module = _build_round(
            _RecordingTarget(),
            _ScaleDroppingEmbedding(embed_scale),
            assistant,
            embed_scale=embed_scale,
        )
        _run_round(module, 2)
        self.assertTrue(
            torch.equal(
                assistant.inputs[0][..., :2], torch.full((1, 1, 2), 10.0 * embed_scale)
            )
        )
        with self.assertRaises(AssertionError):
            self._assert_both_drafts_scaled(assistant, embed_scale)

    def test_non_finite_or_non_positive_scales_are_rejected(self) -> None:
        for embed_scale in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            with self.subTest(embed_scale=embed_scale):
                with self.assertRaisesRegex(ValueError, "target embedding scale"):
                    _build_round(
                        _RecordingTarget(),
                        _TokenEmbedding(),
                        _RecordingAssistant(),
                        embed_scale=embed_scale,
                    )

    def test_degenerate_round_dimensions_are_rejected(self) -> None:
        for hidden_size, max_input_len, max_donor_len in (
            (0, 3, 2),
            (2, 2, 2),
            (2, 3, 1),
        ):
            with self.subTest(hidden_size=hidden_size, max_input_len=max_input_len):
                with self.assertRaisesRegex(ValueError, "combined-round dimensions"):
                    K2GPUResidentRound(
                        _RecordingTarget(),
                        _TokenEmbedding(),
                        _RecordingAssistant(),
                        hidden_size=hidden_size,
                        max_input_len=max_input_len,
                        max_donor_len=max_donor_len,
                        embed_scale=1.0,
                    )


class K2RoundOutputTest(unittest.TestCase):
    """Round vs prefill selection of matches, bonus, feature and the state probe."""

    def test_round_mode_emits_longest_prefix_evidence(self) -> None:
        assistant = _RecordingAssistant()
        module = _build_round(_RecordingTarget(), _TokenEmbedding(), assistant)
        candidates, greedy, matches, bonus, probe = _run_round(module, 2)
        self.assertEqual(candidates.tolist(), [[21, 22]])
        self.assertEqual(candidates.dtype, torch.int64)
        self.assertEqual(greedy.tolist(), [[21, 22, 23]])
        self.assertEqual(matches.tolist(), [2])
        self.assertEqual(bonus.tolist(), [[23]])
        self.assertEqual(probe.dtype, torch.float32)
        self.assertEqual(tuple(probe.shape), (1, 1))
        self.assertEqual(probe.tolist(), [[5.0]])

    def test_rejected_drafts_fall_back_to_the_first_target_row(self) -> None:
        assistant = _RecordingAssistant(first_token=5)
        module = _build_round(_RecordingTarget(), _TokenEmbedding(), assistant)
        candidates, _greedy, matches, bonus, probe = _run_round(module, 2)
        self.assertEqual(candidates.tolist(), [[5, 6]])
        self.assertEqual(matches.tolist(), [0])
        self.assertEqual(bonus.tolist(), [[21]])
        self.assertEqual(probe.tolist(), [[1.0]])

    def test_prefill_mode_zeroes_matches_and_takes_the_last_greedy(self) -> None:
        assistant = _RecordingAssistant(first_token=5)
        module = _build_round(_RecordingTarget(), _TokenEmbedding(), assistant)
        _candidates, _greedy, matches, bonus, probe = _run_round(module, 2, is_round=0)
        self.assertEqual(matches.tolist(), [0])
        self.assertEqual(bonus.tolist(), [[23]])
        self.assertEqual(probe.tolist(), [[5.0]])

    def test_state_probe_reads_the_updated_seed_feature(self) -> None:
        module = _build_round(
            _RecordingTarget(),
            _TokenEmbedding(),
            _RecordingAssistant(),
            round_class=K2GPUResidentRound,
        )
        self.assertEqual(module.seed_feature.tolist(), [[[[0.0, 0.0]]]])
        probe = _run_round(module, 2)[-1]
        self.assertEqual(module.seed_feature.tolist(), [[[[5.0, 6.0]]]])
        self.assertEqual(probe.tolist(), [[5.0]])


class LongestPrefixSelectorTest(unittest.TestCase):
    def test_prefix_length_drives_bonus_and_feature(self) -> None:
        selector = K2LongestPrefixSelector()
        features = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
        greedy = torch.tensor([[90, 91, 92]])

        for drafts, expected_count, expected_bonus, expected_feature in (
            ([80, 81], 0, 90, [10.0, 11.0]),
            ([90, 81], 1, 91, [20.0, 21.0]),
            ([90, 91], 2, 92, [30.0, 31.0]),
            ([80, 91], 0, 90, [10.0, 11.0]),
        ):
            with self.subTest(drafts=drafts):
                count, bonus, candidates, feature = selector(
                    torch.tensor([[7, *drafts]]), greedy, features
                )
                self.assertEqual(count.tolist(), [expected_count])
                self.assertEqual(count.dtype, torch.int64)
                self.assertEqual(bonus.tolist(), [expected_bonus])
                self.assertEqual(candidates.tolist(), [drafts])
                self.assertEqual(feature.tolist(), [expected_feature])


class _NegativeSelectChains(torch.nn.Module):
    def __init__(self, chains: int) -> None:
        super().__init__()
        self.chains = chains

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((1,), dtype=torch.int64)
        for source in (first, second)[: self.chains]:
            total = total + source[-1].item()
        return total


def _export_negative_select_chains(chains: int) -> torch.export.ExportedProgram:
    positions = torch.arange(3, dtype=torch.int64)
    return torch.export.export(
        _NegativeSelectChains(chains), (positions, positions.clone()), strict=False
    )


class SelectAsSymintRewriteTest(unittest.TestCase):
    """`compose_k2_round_program` demands exactly two negative-select rewrites."""

    def test_rewrite_count_tracks_negative_select_chains(self) -> None:
        for chains in (0, 1, 2):
            with self.subTest(chains=chains):
                self.assertEqual(
                    _rewrite_negative_select_as_symint(
                        _export_negative_select_chains(chains)
                    ),
                    chains,
                )


class _TinyRoundBoundFixture(torch.nn.Module):
    def __init__(self, tail_width: int) -> None:
        super().__init__()
        self.register_buffer("round_tail", torch.zeros((1, tail_width)))


class ExportGuardTest(unittest.TestCase):
    def test_round_tail_must_cover_the_declared_input_bound(self) -> None:
        for max_input_len, tail_width in ((2, 0), (8, 4)):
            with self.subTest(max_input_len=max_input_len):
                with self.assertRaisesRegex(ValueError, "round input bound"):
                    export_k2_round_program(
                        # pyre-ignore[6]: the bound guard runs before any module use.
                        _TinyRoundBoundFixture(tail_width),
                        max_input_len,
                    )


class Emb4TargetContractTest(unittest.TestCase):
    """The K=2 target is pinned to `8da4w+emb4`, group size 128 and a 4-bit head."""

    def _build(self, **overrides: object) -> None:
        arguments: dict[str, object] = {
            "max_seq_len": 8960,
            "max_input_len": 512,
            "text_quantize": "8da4w+emb4",
            "assistant_quantize": "8da4w",
            "assistant_lm_head_bits": 4,
            "group_size": 128,
        }
        arguments.update(overrides)
        build_k2_round_program(
            Path("/nonexistent/gemma4-target"),
            Path("/nonexistent/gemma4-assistant"),
            # pyre-ignore[6]: parametrised guard arguments.
            **arguments,
        )

    def test_emb8_and_other_target_quantizations_fail_closed(self) -> None:
        for text_quantize in ("8da4w+emb8", "8da4w", "emb4", "8da4w+emb4 ", ""):
            with self.subTest(text_quantize=text_quantize):
                with self.assertRaisesRegex(ValueError, r"8da4w\+emb4"):
                    self._build(text_quantize=text_quantize)

    def test_group_size_is_pinned_to_128(self) -> None:
        for group_size in (32, 64, 256):
            with self.subTest(group_size=group_size):
                with self.assertRaisesRegex(ValueError, "group size 128"):
                    self._build(group_size=group_size)

    def test_assistant_head_is_pinned_to_four_bits_and_8da4w(self) -> None:
        with self.assertRaisesRegex(ValueError, "4-bit LM head"):
            self._build(assistant_lm_head_bits=8)
        with self.assertRaisesRegex(ValueError, "4-bit LM head"):
            self._build(assistant_quantize="8da8w")

    def test_sequence_bounds_are_guarded(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_seq_len >= 514"):
            self._build(max_seq_len=513)
        with self.assertRaisesRegex(ValueError, "max_input_len"):
            self._build(max_input_len=2)
        with self.assertRaisesRegex(ValueError, "max_input_len"):
            self._build(max_seq_len=1024, max_input_len=1025)

    def test_partitioner_rejects_non_emb4_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "emb4"):
            build_webgpu_partitioner(text_quantize="8da4w+emb8", mode="mtp")

    def test_partitioner_refuses_conflicting_compile_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot override"):
            build_webgpu_partitioner(
                "8da4w+emb4",
                mode="mtp",
                compile_options={"require_dynamic_shapes": False},
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            build_webgpu_partitioner(
                "8da4w+emb4",
                mode="mtp",
                compile_options={"skip_bool_tensors": True},
            )

    def test_lowering_forwards_the_target_quantization_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "emb4"):
            _lower_k2_round(
                _export_negative_select_chains(0),
                external_constants_max_data_bytes=1024,
                text_quantize="8da4w+emb8",
            )


class _TwoResidualChains(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "ordering",
            torch.arange(262144, dtype=torch.float32).reshape(2048, 128),
            persistent=False,
        )
        self.register_buffer("output", torch.zeros((1, 1, 262144), dtype=torch.float32))

    def _chain(self, scores: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(scores, 32, dim=-1)
        destinations = (
            torch.nn.functional.embedding(indices, self.ordering)
            .to(torch.long)
            .view(1, 1, 4096)
        )
        source = torch.ones((1, 1, 4096), dtype=torch.float32)
        return self.output.scatter(-1, destinations, source)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._chain(first), self._chain(second)


class _DisconnectedResidualChains(_TwoResidualChains):
    def _chain(self, scores: torch.Tensor) -> torch.Tensor:
        values, _ = torch.topk(scores, 32, dim=-1)
        destinations = torch.arange(4096).view(1, 1, 4096)
        source = values.repeat_interleave(128, dim=-1)
        return self.output.scatter(-1, destinations, source)


class _TopKOnly(torch.nn.Module):
    def forward(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(scores, 32, dim=-1, largest=True, sorted=True)


class _ScatterOnly(torch.nn.Module):
    def forward(
        self, output: torch.Tensor, index: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        return output.scatter(-1, index, source)


def _find_call(
    program: torch.export.ExportedProgram, target: torch._ops.OpOverload
) -> torch.fx.Node:
    return next(
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == target
    )


def _export_chains(module: torch.nn.Module) -> torch.export.ExportedProgram:
    scores = torch.zeros((1, 1, 2048), dtype=torch.float32)
    return torch.export.export(
        module, (scores, scores.clone()), strict=False
    ).run_decompositions({})


class MtpScatterRewriteTest(unittest.TestCase):
    @unittest.skip("requires the final assistant residual topology")
    def test_rewrite_is_scoped_to_two_certified_chains(self) -> None:
        program = _export_chains(_TwoResidualChains())
        ordering = torch.arange(262144, dtype=torch.int64)
        self.assertEqual(rewrite_certified_unique_scatter(program, ordering), 2)
        targets = [
            node.target for node in program.graph.nodes if node.op == "call_function"
        ]
        self.assertEqual(targets.count(torch.ops.et_vk.scatter_src_unique.default), 2)
        self.assertNotIn(torch.ops.aten.scatter.src, targets)

    def test_non_permutation_ordering_is_rejected(self) -> None:
        duplicate = torch.arange(262144, dtype=torch.int64)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "permutation"):
            rewrite_certified_unique_scatter(
                _export_chains(_TwoResidualChains()), duplicate
            )

    @unittest.skip("requires the final assistant residual topology")
    def test_ordering_provenance_must_match_the_baked_buffer(self) -> None:
        ordering = torch.arange(262144, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "token-ordering conversion mismatch"):
            rewrite_certified_unique_scatter(
                _export_chains(_TwoResidualChains()), ordering.flip(0)
            )

    def test_scatter_without_a_topk_ancestor_is_rejected(self) -> None:
        ordering = torch.arange(262144, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "token-ordering conversion mismatch"):
            rewrite_certified_unique_scatter(
                _export_chains(_DisconnectedResidualChains()), ordering
            )

    def test_chain_count_other_than_two_is_rejected(self) -> None:
        ordering = torch.arange(262144, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "residual topology mismatch"):
            rewrite_certified_unique_scatter(
                _export_chains(_TwoResidualChains()), ordering, expected_chains=3
            )
        with self.assertRaisesRegex(ValueError, "residual topology mismatch"):
            rewrite_certified_unique_scatter(
                _export_chains(_TwoResidualChains()), ordering, expected_chains=0
            )


class MtpOpFeatureTest(unittest.TestCase):
    def test_features_are_instance_scoped_and_exact(self) -> None:
        from executorch.backends.vulkan.op_registry import vulkan_supported_ops

        before = dict(vulkan_supported_ops)
        features = mtp_extra_op_features()
        self.assertIn(exir_ops.edge.aten.topk.default, features)
        self.assertIn(exir_ops.edge.et_vk.scatter_src_unique.default, features)
        self.assertEqual(before, vulkan_supported_ops)

    def test_residual_routes_require_exact_full_shapes(self) -> None:
        exact_topk = torch.export.export(
            _TopKOnly(),
            (torch.zeros((1, 1, 2048), dtype=torch.float32),),
            strict=False,
        )
        self.assertTrue(
            _is_official_qat_topk(_find_call(exact_topk, torch.ops.aten.topk.default))
        )
        wrong_rank_topk = torch.export.export(
            _TopKOnly(),
            (torch.zeros((1, 2048), dtype=torch.float32),),
            strict=False,
        )
        self.assertFalse(
            _is_official_qat_topk(
                _find_call(wrong_rank_topk, torch.ops.aten.topk.default)
            )
        )

        exact_scatter = torch.export.export(
            _ScatterOnly(),
            (
                torch.zeros((1, 1, 262144), dtype=torch.float32),
                torch.arange(4096, dtype=torch.int64).reshape(1, 1, 4096),
                torch.ones((1, 1, 4096), dtype=torch.float32),
            ),
            strict=False,
        )
        self.assertTrue(
            _is_official_qat_unique_scatter(
                _find_call(exact_scatter, torch.ops.aten.scatter.src)
            )
        )
        wrong_rank_scatter = torch.export.export(
            _ScatterOnly(),
            (
                torch.zeros((1, 262144), dtype=torch.float32),
                torch.arange(4096, dtype=torch.int64).reshape(1, 4096),
                torch.ones((1, 4096), dtype=torch.float32),
            ),
            strict=False,
        )
        self.assertFalse(
            _is_official_qat_unique_scatter(
                _find_call(wrong_rank_scatter, torch.ops.aten.scatter.src)
            )
        )
