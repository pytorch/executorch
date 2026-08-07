# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import dataclasses
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

import torch

from executorch.backends.vulkan.patterns.rope_hf import HfRotaryEmbeddingSinglePattern

from executorch.examples.models.gemma4.eagle_webgpu_round import (
    OFFICIAL_QAT_CENTROID_TOP_K,
    OFFICIAL_QAT_NUM_CENTROIDS,
    OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
    OFFICIAL_QAT_TOKENS_PER_CENTROID,
    select_qat_centroids,
    validate_qat_token_ordering,
    validate_selected_destinations,
)
from executorch.examples.models.gemma4.export_assistant_webgpu_artifacts import (
    _StaticAssistantQueryRopeLayer,
    _UnusedAssistantRotaryEmbedding,
    adapt_assistant_model_for_webgpu,
    adapt_masked_embedding_for_webgpu,
    QAT_VALIDATION_DONOR_SEQUENCE,
    StaticAssistantMasks,
    StaticAssistantQueryRope,
    StaticAssistantSharedKVAttention,
    UnfoldedAssistant,
    validate_qat_centroid_scores,
    validate_qat_selection_contract,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    ASSISTANT_CHECKPOINT_ACQUISITION,
    ASSISTANT_MODEL_CONTRACT,
)


_VOCAB_SIZE = OFFICIAL_QAT_NUM_CENTROIDS * OFFICIAL_QAT_TOKENS_PER_CENTROID
_SLIDING_WINDOW = 512
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_HUB_REPOSITORY = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+\Z")
_FORBIDDEN_IDENTITY_MARKERS = (
    "/data/",  # oss-closure-fixture
    "/home/",  # oss-closure-fixture
    "/mnt/",  # oss-closure-fixture
    "fbsource",
    "manifold",  # oss-closure-fixture
    "everstore",
    "://",
    "~",
    "..",
    " ",
)


class AssistantIdentityTest(unittest.TestCase):
    """The pinned assistant checkpoint is a public hub coordinate plus digests."""

    def test_repository_and_revision_are_pinned(self) -> None:
        self.assertEqual(
            ASSISTANT_CHECKPOINT_ACQUISITION["repo_id"],
            "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
        )
        self.assertEqual(
            ASSISTANT_CHECKPOINT_ACQUISITION["revision"],
            "ebc7e1a211354561464cb82ed6d886792138dcb6",
        )
        self.assertRegex(str(ASSISTANT_CHECKPOINT_ACQUISITION["revision"]), _HEX40)
        self.assertRegex(
            str(ASSISTANT_CHECKPOINT_ACQUISITION["repo_id"]), _HUB_REPOSITORY
        )

    def test_every_pinned_digest_is_a_sha256(self) -> None:
        files = ASSISTANT_CHECKPOINT_ACQUISITION["files"]
        self.assertIsInstance(files, dict)
        assert isinstance(files, dict)
        self.assertEqual(set(files), {"config.json", "model.safetensors"})
        digests = set()
        for name, identity in files.items():
            with self.subTest(name=name):
                self.assertIsInstance(identity, dict)
                assert isinstance(identity, dict)
                self.assertRegex(str(identity["sha256"]), _HEX64)
                self.assertGreater(int(identity["bytes"]), 0)
                digests.add(identity["sha256"])
        self.assertEqual(len(digests), 2)

    def test_no_internal_path_or_receipt_is_pinned(self) -> None:
        value = json.dumps(
            ASSISTANT_CHECKPOINT_ACQUISITION,
            sort_keys=True,
            separators=(",", ":"),
        ).lower()
        for marker in _FORBIDDEN_IDENTITY_MARKERS:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, value)

    def test_model_contract_is_the_official_assistant_shape(self) -> None:
        self.assertEqual(
            ASSISTANT_MODEL_CONTRACT,
            {
                "architecture": "Gemma4AssistantForCausalLM",
                "backboneHiddenSize": 1536,
                "hiddenSize": 256,
                "modelType": "gemma4_assistant",
                "numHiddenLayers": 4,
                "vocabSize": 262144,
            },
        )

    def test_validation_donor_sequence_brackets_the_sliding_window(self) -> None:
        self.assertEqual(
            QAT_VALIDATION_DONOR_SEQUENCE,
            (2, 16, 511, 512, 513, 514, 1024, 8960, 2),
        )
        self.assertEqual(QAT_VALIDATION_DONOR_SEQUENCE[0], 2)
        self.assertEqual(QAT_VALIDATION_DONOR_SEQUENCE[-1], 2)
        for boundary in (_SLIDING_WINDOW - 1, _SLIDING_WINDOW, _SLIDING_WINDOW + 1):
            self.assertIn(boundary, QAT_VALIDATION_DONOR_SEQUENCE)


class QatSelectionDimensionsTest(unittest.TestCase):
    """2048 centroids x 128 tokens, 32 selected centroids, 4096 selected tokens."""

    def test_official_selection_dimensions(self) -> None:
        self.assertEqual(OFFICIAL_QAT_NUM_CENTROIDS, 2048)
        self.assertEqual(OFFICIAL_QAT_TOKENS_PER_CENTROID, 128)
        self.assertEqual(OFFICIAL_QAT_CENTROID_TOP_K, 32)
        self.assertEqual(OFFICIAL_QAT_SELECTED_TOKEN_COUNT, 4096)
        self.assertEqual(
            OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
            OFFICIAL_QAT_CENTROID_TOP_K * OFFICIAL_QAT_TOKENS_PER_CENTROID,
        )
        self.assertEqual(_VOCAB_SIZE, ASSISTANT_MODEL_CONTRACT["vocabSize"])


def _token_ordering(seed: int = 0xE4A6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(_VOCAB_SIZE, generator=generator, dtype=torch.int64)


class QatTokenOrderingTest(unittest.TestCase):
    def test_raw_and_logical_orderings_produce_the_same_evidence(self) -> None:
        raw = _token_ordering()
        logical = raw.reshape(
            OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID
        )
        raw_evidence = validate_qat_token_ordering(raw)
        logical_evidence = validate_qat_token_ordering(logical)
        self.assertEqual(raw_evidence["rawShape"], [_VOCAB_SIZE])
        self.assertEqual(logical_evidence["rawShape"], [2048, 128])
        self.assertEqual(raw_evidence["shape"], [2048, 128])
        self.assertEqual(logical_evidence["shape"], [2048, 128])
        for key in ("max", "min", "numel", "sha256", "uniqueCount", "permutationExact"):
            with self.subTest(key=key):
                self.assertEqual(raw_evidence[key], logical_evidence[key])
        self.assertEqual(raw_evidence["min"], 0)
        self.assertEqual(raw_evidence["max"], _VOCAB_SIZE - 1)
        self.assertEqual(raw_evidence["numel"], _VOCAB_SIZE)
        self.assertEqual(raw_evidence["uniqueCount"], _VOCAB_SIZE)
        self.assertIs(raw_evidence["permutationExact"], True)

    def test_a_single_duplicated_entry_is_rejected(self) -> None:
        duplicate = _token_ordering()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            validate_qat_token_ordering(duplicate)

    def test_shifted_ordering_outside_the_vocabulary_is_rejected(self) -> None:
        shifted = _token_ordering() + 1
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            validate_qat_token_ordering(shifted)

    def test_wrong_element_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "262144 entries"):
            validate_qat_token_ordering(torch.arange(_VOCAB_SIZE - 1))

    def test_non_integer_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            validate_qat_token_ordering(torch.arange(_VOCAB_SIZE, dtype=torch.float32))


class QatSelectedDestinationTest(unittest.TestCase):
    def test_thirty_two_centroids_map_to_4096_distinct_destinations(self) -> None:
        ordering = _token_ordering()
        scores = torch.arange(OFFICIAL_QAT_NUM_CENTROIDS, dtype=torch.float32).reshape(
            1, 1, OFFICIAL_QAT_NUM_CENTROIDS
        )
        selected = select_qat_centroids(scores)
        self.assertEqual(tuple(selected.shape), (OFFICIAL_QAT_CENTROID_TOP_K,))
        highest = OFFICIAL_QAT_NUM_CENTROIDS - 1
        self.assertEqual(
            selected.tolist(),
            list(range(highest, highest - OFFICIAL_QAT_CENTROID_TOP_K, -1)),
        )
        destinations = validate_selected_destinations(ordering, selected)
        self.assertEqual(destinations.numel(), OFFICIAL_QAT_SELECTED_TOKEN_COUNT)
        self.assertEqual(torch.unique(destinations).numel(), destinations.numel())
        logical = ordering.reshape(
            OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID
        )
        self.assertTrue(
            torch.equal(destinations, logical[selected].reshape(-1)),
        )

    def test_duplicate_selected_centroids_are_rejected(self) -> None:
        ordering = _token_ordering()
        selected = torch.arange(OFFICIAL_QAT_CENTROID_TOP_K, dtype=torch.int64)
        selected[-1] = selected[0]
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            validate_selected_destinations(ordering, selected)

    def test_out_of_range_and_miscounted_centroids_are_rejected(self) -> None:
        ordering = _token_ordering()
        too_large = torch.arange(OFFICIAL_QAT_CENTROID_TOP_K, dtype=torch.int64)
        too_large[0] = OFFICIAL_QAT_NUM_CENTROIDS
        with self.assertRaisesRegex(ValueError, "index out of range"):
            validate_selected_destinations(ordering, too_large)
        negative = torch.arange(OFFICIAL_QAT_CENTROID_TOP_K, dtype=torch.int64)
        negative[0] = -1
        with self.assertRaisesRegex(ValueError, "index out of range"):
            validate_selected_destinations(ordering, negative)
        with self.assertRaisesRegex(ValueError, "32 centroids"):
            validate_selected_destinations(
                ordering, torch.arange(31, dtype=torch.int64)
            )

    def test_non_permutation_ordering_is_rejected_before_selection(self) -> None:
        duplicate = _token_ordering()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            validate_selected_destinations(
                duplicate, torch.arange(OFFICIAL_QAT_CENTROID_TOP_K, dtype=torch.int64)
            )

    def test_centroid_score_count_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "2048 entries"):
            select_qat_centroids(torch.zeros(2047, dtype=torch.float32))


class QatCentroidScoreTest(unittest.TestCase):
    def _scores(self) -> torch.Tensor:
        return torch.arange(OFFICIAL_QAT_NUM_CENTROIDS, dtype=torch.float32).reshape(
            1, 1, OFFICIAL_QAT_NUM_CENTROIDS
        )

    def test_strictly_ordered_scores_produce_a_positive_boundary_gap(self) -> None:
        evidence = validate_qat_centroid_scores(self._scores())
        self.assertIs(evidence["allFinite"], True)
        self.assertIs(evidence["top32PairwiseDistinct"], True)
        self.assertEqual(evidence["boundaryGap"], 1.0)
        for key in (
            "indicesSha256",
            "top33IndicesSha256",
            "top33ValuesSha256",
            "valuesSha256",
        ):
            with self.subTest(key=key):
                self.assertRegex(str(evidence[key]), _HEX64)

    def test_a_tie_at_the_32_33_boundary_is_rejected(self) -> None:
        scores = self._scores()
        scores[..., OFFICIAL_QAT_NUM_CENTROIDS - 33] = scores[
            ..., OFFICIAL_QAT_NUM_CENTROIDS - 32
        ]
        with self.assertRaisesRegex(ValueError, "boundary gap"):
            validate_qat_centroid_scores(scores)

    def test_a_tie_inside_the_top_32_is_rejected(self) -> None:
        scores = self._scores()
        scores[..., OFFICIAL_QAT_NUM_CENTROIDS - 2] = scores[
            ..., OFFICIAL_QAT_NUM_CENTROIDS - 1
        ]
        with self.assertRaisesRegex(ValueError, "pairwise distinct"):
            validate_qat_centroid_scores(scores)

    def test_non_finite_scores_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                scores = self._scores()
                scores[..., 0] = bad
                with self.assertRaisesRegex(ValueError, "finite fp32"):
                    validate_qat_centroid_scores(scores)

    def test_shape_and_dtype_are_pinned(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \[1, 1, 2048\]"):
            validate_qat_centroid_scores(
                torch.zeros((1, OFFICIAL_QAT_NUM_CENTROIDS), dtype=torch.float32)
            )
        with self.assertRaisesRegex(ValueError, "finite fp32"):
            validate_qat_centroid_scores(
                torch.zeros((1, 1, OFFICIAL_QAT_NUM_CENTROIDS), dtype=torch.float64)
            )


class _MaskedEmbeddingFixture(torch.nn.Module):
    """Minimal stand-in for the QAT masked embedding head."""

    def __init__(self, hidden_size: int = 4, *, with_lm_embed: bool = True) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.centroids = torch.nn.Linear(
            hidden_size, OFFICIAL_QAT_NUM_CENTROIDS, bias=False
        )
        with torch.no_grad():
            self.centroids.weight.copy_(
                torch.randn(
                    (OFFICIAL_QAT_NUM_CENTROIDS, hidden_size), generator=generator
                )
            )
        self.register_buffer("token_ordering", _token_ordering(), persistent=False)
        if with_lm_embed:
            embed = torch.nn.Embedding(_VOCAB_SIZE, hidden_size)
            with torch.no_grad():
                embed.weight.copy_(
                    torch.randn((_VOCAB_SIZE, hidden_size), generator=generator)
                )
            self._lm_embed = embed

    def forward(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        del lm_head_weight
        return self.centroids(hidden_states)


class MaskedEmbeddingAdaptationTest(unittest.TestCase):
    """The static head must reproduce a masked full-vocabulary logit row."""

    def test_static_head_matches_the_dense_masked_reference(self) -> None:
        head = _MaskedEmbeddingFixture()
        ordering = head.token_ordering.clone()
        adapt_masked_embedding_for_webgpu(head)
        generator = torch.Generator().manual_seed(5)
        hidden = torch.randn((1, 1, 4), generator=generator)

        produced = head(hidden, torch.empty(0))
        centroid_logits = head.centroids(hidden)
        top_k = torch.topk(centroid_logits, OFFICIAL_QAT_CENTROID_TOP_K, dim=-1).indices
        selected = (
            ordering.reshape(
                OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID
            )[top_k.reshape(-1)]
            .reshape(1, 1, OFFICIAL_QAT_SELECTED_TOKEN_COUNT)
            .to(torch.int64)
        )
        dense = torch.nn.functional.linear(hidden, head._lm_embed.weight)
        reference = torch.full_like(dense, torch.finfo(torch.float32).min)
        reference.scatter_(-1, selected, dense.gather(-1, selected))

        self.assertEqual(tuple(produced.shape), (1, 1, _VOCAB_SIZE))
        self.assertEqual(produced.dtype, torch.float32)
        torch.testing.assert_close(produced, reference, atol=1e-5, rtol=1e-5)

    def test_unselected_tokens_stay_masked_out(self) -> None:
        head = _MaskedEmbeddingFixture()
        adapt_masked_embedding_for_webgpu(head)
        generator = torch.Generator().manual_seed(6)
        produced = head(torch.randn((1, 1, 4), generator=generator), torch.empty(0))
        masked = produced == torch.finfo(torch.float32).min
        self.assertEqual(int((~masked).sum().item()), OFFICIAL_QAT_SELECTED_TOKEN_COUNT)

    def test_adaptation_publishes_the_static_selection_buffers(self) -> None:
        head = _MaskedEmbeddingFixture()
        ordering = head.token_ordering.clone()
        adapt_masked_embedding_for_webgpu(head)
        self.assertEqual(
            tuple(head._webgpu_token_ordering.shape),
            (OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID),
        )
        self.assertEqual(head._webgpu_token_ordering.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                head._webgpu_token_ordering.to(torch.int64).reshape(-1), ordering
            )
        )
        self.assertEqual(tuple(head._webgpu_output_template.shape), (1, 1, _VOCAB_SIZE))
        self.assertTrue(
            torch.all(head._webgpu_output_template == torch.finfo(torch.float32).min)
        )

    def test_adaptation_requires_a_quantized_lm_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantized LM head"):
            adapt_masked_embedding_for_webgpu(
                _MaskedEmbeddingFixture(with_lm_embed=False)
            )


class StaticAssistantMaskTest(unittest.TestCase):
    """The sliding mask exposes the newest `sliding_window + 1` donor rows."""

    def test_masks_track_the_donor_window(self) -> None:
        masks = StaticAssistantMasks(max_seq_len=8960)
        blocked = torch.finfo(torch.float32).min
        for donor_length in (2, 511, 512, 513, 514, 8960):
            with self.subTest(donor_length=donor_length):
                donor = torch.zeros((1, 1, donor_length, 1))
                full, sliding = masks(donor, donor)
                self.assertEqual(tuple(full.shape), (1, donor_length))
                self.assertEqual(tuple(sliding.shape), (1, donor_length))
                self.assertTrue(torch.equal(full, torch.zeros_like(full)))
                visible = min(donor_length, _SLIDING_WINDOW + 1)
                self.assertTrue(
                    torch.equal(sliding[:, -visible:], torch.zeros((1, visible)))
                )
                if donor_length > visible:
                    self.assertTrue(
                        torch.equal(
                            sliding[:, :-visible],
                            torch.full((1, donor_length - visible), blocked),
                        )
                    )

    def test_full_and_sliding_donors_may_have_different_lengths(self) -> None:
        masks = StaticAssistantMasks(max_seq_len=4096)
        full, sliding = masks(torch.zeros((1, 1, 4096, 1)), torch.zeros((1, 1, 600, 1)))
        self.assertEqual(tuple(full.shape), (1, 4096))
        self.assertEqual(tuple(sliding.shape), (1, 600))

    def test_degenerate_capacities_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention-mask capacity"):
            StaticAssistantMasks(max_seq_len=1)
        with self.assertRaisesRegex(ValueError, "attention-mask capacity"):
            StaticAssistantMasks(max_seq_len=1024, sliding_window=0)


class _AssistantRotaryEmbedding(torch.nn.Module):
    """Per-layer-type rotary table with the HF rotate-half layout."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        half = head_dim // 2
        exponent = torch.arange(0, half, dtype=torch.float32) / half
        self.register_buffer(
            "full_inv_freq", 1.0 / (10000.0**exponent), persistent=False
        )
        self.register_buffer(
            "sliding_inv_freq", 1.0 / (100.0**exponent), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = (
            self.sliding_inv_freq
            if layer_type == "sliding_attention"
            else self.full_inv_freq
        )
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
        table = torch.cat((angles, angles), dim=-1)
        return table.cos().to(x.dtype), table.sin().to(x.dtype)


class _ReferenceSharedKVAttention(torch.nn.Module):
    """Unadapted layer: rotary tables recomputed per call, plain fp32 SDPA."""

    def __init__(
        self,
        rotary: _AssistantRotaryEmbedding,
        *,
        layer_type: str,
        hidden_size: int = 8,
        num_attention_heads: int = 2,
        head_dim: int = 4,
        is_kv_shared_layer: bool = True,
    ) -> None:
        super().__init__()
        self.is_kv_shared_layer = is_kv_shared_layer
        self.layer_type = layer_type
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.q_proj = torch.nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.q_norm = torch.nn.LayerNorm(head_dim)
        self.o_proj = torch.nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )
        self.rotary = rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).view(
            *input_shape, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query)
        cos, sin = self.rotary(query, position_ids, self.layer_type)
        query = HfRotaryEmbeddingSinglePattern()(query, cos.squeeze(0), sin.squeeze(0))
        key, value = shared_kv_states[self.layer_type]
        attention = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key, value, attn_mask=attention_mask, scale=1.0
        ).transpose(1, 2)
        return self.o_proj(attention.reshape(*input_shape, -1))


def _donor_states(
    donor_length: int, heads: int = 2, head_dim: int = 4, seed: int = 3
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_type in ("full_attention", "sliding_attention"):
        states[layer_type] = (
            torch.randn((1, heads, donor_length, head_dim), generator=generator),
            torch.randn((1, heads, donor_length, head_dim), generator=generator),
        )
    return states


class AdaptedAssistantAttentionTest(unittest.TestCase):
    """`StaticAssistant*` must reproduce the unadapted eager attention output."""

    MAX_SEQ_LEN = 1024
    TOLERANCE = 1e-4

    def _adapted(
        self, reference: _ReferenceSharedKVAttention, rotary: _AssistantRotaryEmbedding
    ) -> StaticAssistantSharedKVAttention:
        tables = StaticAssistantQueryRope(rotary, max_seq_len=self.MAX_SEQ_LEN)
        source = getattr(tables, reference.layer_type)
        return StaticAssistantSharedKVAttention(
            reference,
            _StaticAssistantQueryRopeLayer(source.freqs_cos, source.freqs_sin),
        )

    def test_precomputed_rope_tables_match_the_live_rotary_module(self) -> None:
        torch.manual_seed(0)
        rotary = _AssistantRotaryEmbedding(head_dim=4)
        tables = StaticAssistantQueryRope(rotary, max_seq_len=self.MAX_SEQ_LEN)
        query = torch.randn((1, 1, 2, 4))
        for layer_type in ("full_attention", "sliding_attention"):
            for position in (0, 1, 511, 512, 513, self.MAX_SEQ_LEN - 1):
                with self.subTest(layer_type=layer_type, position=position):
                    position_ids = torch.tensor([[position]], dtype=torch.int64)
                    cos, sin = rotary(query, position_ids, layer_type)
                    reference = HfRotaryEmbeddingSinglePattern()(
                        query, cos.squeeze(0), sin.squeeze(0)
                    )
                    source = getattr(tables, layer_type)
                    produced = _StaticAssistantQueryRopeLayer(
                        source.freqs_cos, source.freqs_sin
                    )(query, position_ids)
                    torch.testing.assert_close(
                        produced, reference, atol=self.TOLERANCE, rtol=self.TOLERANCE
                    )

    def test_rope_tables_are_layer_type_specific(self) -> None:
        torch.manual_seed(0)
        tables = StaticAssistantQueryRope(
            _AssistantRotaryEmbedding(head_dim=4), max_seq_len=self.MAX_SEQ_LEN
        )
        self.assertFalse(
            torch.equal(
                tables.full_attention.freqs_cos, tables.sliding_attention.freqs_cos
            )
        )
        self.assertEqual(
            tuple(tables.full_attention.freqs_cos.shape), (self.MAX_SEQ_LEN, 4)
        )

    def test_adapted_layer_matches_the_unadapted_layer(self) -> None:
        torch.manual_seed(0)
        rotary = _AssistantRotaryEmbedding(head_dim=4)
        masks = StaticAssistantMasks(self.MAX_SEQ_LEN)
        hidden = torch.randn((1, 1, 8))
        for layer_type in ("full_attention", "sliding_attention"):
            for donor_length in (2, 511, 512, 513, 514, self.MAX_SEQ_LEN):
                with self.subTest(layer_type=layer_type, donor_length=donor_length):
                    reference = _ReferenceSharedKVAttention(
                        rotary, layer_type=layer_type
                    )
                    adapted = self._adapted(reference, rotary)
                    shared = _donor_states(donor_length)
                    full_mask, sliding_mask = masks(
                        shared["full_attention"][0], shared["sliding_attention"][0]
                    )
                    mask = full_mask if layer_type == "full_attention" else sliding_mask
                    position_ids = torch.tensor([[donor_length - 1]], dtype=torch.int64)
                    expected = reference(hidden, mask, shared, position_ids)
                    produced, extra = adapted(hidden, None, mask, shared, position_ids)
                    self.assertIsNone(extra)
                    self.assertEqual(produced.shape, expected.shape)
                    torch.testing.assert_close(
                        produced, expected, atol=self.TOLERANCE, rtol=self.TOLERANCE
                    )

    def test_adapted_layer_reads_only_its_own_shared_kv_entry(self) -> None:
        torch.manual_seed(0)
        rotary = _AssistantRotaryEmbedding(head_dim=4)
        reference = _ReferenceSharedKVAttention(rotary, layer_type="full_attention")
        adapted = self._adapted(reference, rotary)
        hidden = torch.randn((1, 1, 8))
        mask = torch.zeros((1, 8))
        position_ids = torch.tensor([[7]], dtype=torch.int64)
        shared = _donor_states(8)
        baseline, _ = adapted(hidden, None, mask, shared, position_ids)
        swapped = dict(shared)
        swapped["sliding_attention"] = _donor_states(8, seed=99)["sliding_attention"]
        produced, _ = adapted(hidden, None, mask, swapped, position_ids)
        torch.testing.assert_close(
            produced, baseline, atol=self.TOLERANCE, rtol=self.TOLERANCE
        )

    def test_a_layer_without_shared_target_kv_is_rejected(self) -> None:
        torch.manual_seed(0)
        rotary = _AssistantRotaryEmbedding(head_dim=4)
        reference = _ReferenceSharedKVAttention(
            rotary, layer_type="full_attention", is_kv_shared_layer=False
        )
        tables = StaticAssistantQueryRope(rotary, max_seq_len=self.MAX_SEQ_LEN)
        with self.assertRaisesRegex(ValueError, "shared target KV"):
            StaticAssistantSharedKVAttention(reference, tables.full_attention)

    def test_a_query_beyond_the_rope_table_is_rejected(self) -> None:
        torch.manual_seed(0)
        rotary = _AssistantRotaryEmbedding(head_dim=4)
        tables = StaticAssistantQueryRope(rotary, max_seq_len=self.MAX_SEQ_LEN)
        layer = _StaticAssistantQueryRopeLayer(
            tables.full_attention.freqs_cos, tables.full_attention.freqs_sin
        )
        query = torch.randn((1, 1, 2, 4))
        with self.assertRaises(RuntimeError):
            layer(query, torch.tensor([[self.MAX_SEQ_LEN]], dtype=torch.int64))
        with self.assertRaises(RuntimeError):
            layer(query, torch.tensor([[-1]], dtype=torch.int64))


@dataclasses.dataclass
class _AssistantOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class _RoutingAssistant(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> _AssistantOutput:
        self.calls.append(
            {
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "shared_kv_states": shared_kv_states,
                "use_cache": use_cache,
            }
        )
        return _AssistantOutput(
            logits=torch.full((1, 1, 3), 7.0),
            last_hidden_state=torch.full((1, 1, 2), 9.0),
        )


class UnfoldedAssistantRoutingTest(unittest.TestCase):
    def test_donor_tensors_are_routed_to_their_layer_types(self) -> None:
        inner = _RoutingAssistant()
        wrapper = UnfoldedAssistant(inner)
        embeds = torch.zeros((1, 1, 4))
        position_ids = torch.tensor([[5]], dtype=torch.int64)
        full_k, full_v, sliding_k, sliding_v = (
            torch.full((1, 2, 3, 4), float(marker)) for marker in range(4)
        )

        logits, hidden = wrapper(
            embeds, position_ids, full_k, full_v, sliding_k, sliding_v
        )

        self.assertEqual(len(inner.calls), 1)
        call = inner.calls[0]
        self.assertIsNone(call["attention_mask"])
        self.assertIs(call["use_cache"], False)
        self.assertIs(call["inputs_embeds"], embeds)
        self.assertIs(call["position_ids"], position_ids)
        self.assertEqual(
            sorted(call["shared_kv_states"]), ["full_attention", "sliding_attention"]
        )
        self.assertIs(call["shared_kv_states"]["full_attention"][0], full_k)
        self.assertIs(call["shared_kv_states"]["full_attention"][1], full_v)
        self.assertIs(call["shared_kv_states"]["sliding_attention"][0], sliding_k)
        self.assertIs(call["shared_kv_states"]["sliding_attention"][1], sliding_v)
        self.assertEqual(logits.tolist(), torch.full((1, 1, 3), 7.0).tolist())
        self.assertEqual(hidden.tolist(), torch.full((1, 1, 2), 9.0).tolist())


class _AssistantLayer(torch.nn.Module):
    def __init__(self, self_attn: _ReferenceSharedKVAttention) -> None:
        super().__init__()
        self.self_attn = self_attn


class _AssistantBackbone(torch.nn.Module):
    def __init__(
        self, rotary: _AssistantRotaryEmbedding, layer_types: list[str]
    ) -> None:
        super().__init__()
        self.rotary_emb = rotary
        self.layers = torch.nn.ModuleList(
            [
                _AssistantLayer(_ReferenceSharedKVAttention(rotary, layer_type=name))
                for name in layer_types
            ]
        )


class _AssistantModel(torch.nn.Module):
    def __init__(self, layer_types: list[str] | None = None) -> None:
        super().__init__()
        types = layer_types or [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.model = _AssistantBackbone(_AssistantRotaryEmbedding(head_dim=4), types)
        self.masked_embedding = _MaskedEmbeddingFixture()


class AssistantAdaptationTest(unittest.TestCase):
    MAX_SEQ_LEN = 1024

    def test_adaptation_rewires_every_layer_and_the_masked_head(self) -> None:
        torch.manual_seed(0)
        assistant = _AssistantModel()
        adapt_assistant_model_for_webgpu(assistant, max_seq_len=self.MAX_SEQ_LEN)
        self.assertIsInstance(
            assistant.model.rotary_emb, _UnusedAssistantRotaryEmbedding
        )
        self.assertIsInstance(assistant._webgpu_static_masks, StaticAssistantMasks)
        self.assertEqual(assistant._webgpu_static_masks.max_seq_len, self.MAX_SEQ_LEN)
        self.assertEqual(
            [layer.self_attn.layer_type for layer in assistant.model.layers],
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        for layer in assistant.model.layers:
            self.assertIsInstance(layer.self_attn, StaticAssistantSharedKVAttention)
        self.assertTrue(hasattr(assistant.masked_embedding, "_webgpu_token_ordering"))

    def test_unexpected_layer_types_are_rejected(self) -> None:
        torch.manual_seed(0)
        sliding = "sliding_attention"
        for layer_types in (
            ["full_attention", sliding, sliding, sliding],
            [sliding, sliding, "full_attention"],
        ):
            with self.subTest(layer_types=layer_types):
                with self.assertRaisesRegex(ValueError, "assistant layer types"):
                    adapt_assistant_model_for_webgpu(
                        _AssistantModel(layer_types), max_seq_len=self.MAX_SEQ_LEN
                    )

    def test_static_masks_replace_the_caller_supplied_attention_mask(self) -> None:
        torch.manual_seed(0)
        assistant = _AssistantModel()
        adapt_assistant_model_for_webgpu(assistant, max_seq_len=self.MAX_SEQ_LEN)
        shared = _donor_states(600)
        masks = assistant.create_attention_masks(torch.zeros((1, 1, 8)), None, shared)
        self.assertEqual(sorted(masks), ["full_attention", "sliding_attention"])
        self.assertEqual(tuple(masks["full_attention"].shape), (1, 600))
        self.assertEqual(tuple(masks["sliding_attention"].shape), (1, 600))
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            assistant.create_attention_masks(
                torch.zeros((1, 1, 8)), torch.zeros((1, 600)), shared
            )


class _QatTextConfig:
    global_head_dim = 1
    head_dim = 1


class _QatAssistantConfig:
    backbone_hidden_size = 1

    def get_text_config(self) -> _QatTextConfig:
        return _QatTextConfig()


class _QatCentroidHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.centroids = torch.nn.Linear(1, OFFICIAL_QAT_NUM_CENTROIDS, bias=False)
        with torch.no_grad():
            self.centroids.weight.copy_(
                torch.randn((OFFICIAL_QAT_NUM_CENTROIDS, 1), generator=generator)
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.centroids(hidden)


class _QatValidationAssistant(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _QatAssistantConfig()
        self.masked_embedding = _QatCentroidHead()


class _QatValidationWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.assistant = _QatValidationAssistant()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.assistant.masked_embedding(inputs[0][..., :1])


class QatSelectionContractTest(unittest.TestCase):
    """The QAT receipt replays donor length 2 and brackets the window boundaries."""

    def test_receipt_covers_the_documented_donor_axes(self) -> None:
        evidence = validate_qat_selection_contract(
            # pyre-ignore[6]: the wrapper duck-types `UnfoldedAssistant`.
            _QatValidationWrapper(),
            max_donor_len=8960,
        )
        self.assertEqual(
            evidence["donorSequence"], [2, 16, 511, 512, 513, 514, 1024, 8960, 2]
        )
        self.assertEqual(
            evidence["selectionContract"],
            {
                "centroidTopK": 32,
                "numCentroids": 2048,
                "selectedTokenCount": 4096,
                "tokensPerCentroid": 128,
            },
        )
        cases = evidence["cases"]
        self.assertEqual(len(cases), 9)
        self.assertEqual([case["caseIndex"] for case in cases], list(range(9)))
        self.assertEqual(cases[0]["inputSha256"], cases[-1]["inputSha256"])
        self.assertEqual(cases[0]["topk"], cases[-1]["topk"])
        self.assertNotEqual(cases[0]["inputSha256"], cases[1]["inputSha256"])

    def test_short_capacity_truncates_the_sequence_but_keeps_the_replay(self) -> None:
        evidence = validate_qat_selection_contract(
            # pyre-ignore[6]: the wrapper duck-types `UnfoldedAssistant`.
            _QatValidationWrapper(),
            max_donor_len=512,
        )
        self.assertEqual(evidence["donorSequence"], [2, 16, 511, 512, 2])

    def test_capacity_below_the_replay_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay at donor length 2"):
            validate_qat_selection_contract(
                # pyre-ignore[6]: the wrapper duck-types `UnfoldedAssistant`.
                _QatValidationWrapper(),
                max_donor_len=1,
            )


class _FakeExportGraphModule:
    def __init__(self) -> None:
        self.meta: dict[str, object] = {
            "gemma4K2Abi": {"fixture": "k2_abi"},
            "gemma4QATSelectionEvidence": {"fixture": "qat_selection"},
            "gemma4TargetCheckpointEvidence": {"fixture": "target_checkpoint"},
        }


class _FakeK2Program:
    def __init__(self) -> None:
        self.graph_module = _FakeExportGraphModule()


class _FakeExecutorchProgram:
    def __init__(self) -> None:
        self._tensor_data = {f"constants_{index}": object() for index in range(3)}

    def write_to_file(self, output: Any) -> None:
        output.write(b"pte")

    def write_tensor_data_to_file(self, directory: str) -> None:
        root = Path(directory)
        for tag in self._tensor_data:
            (root / f"{tag}.ptd").write_bytes(tag.encode("utf-8"))


class SpeculativeExportPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        from executorch.examples.models.gemma4 import export_speculative

        self.export_module = export_speculative
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.output_root = self.root / "sealed"
        self.output = self.output_root / "model.pte"
        self.receipt = self.root / "receipts" / "manifest.json"
        self.source = self.root / "source" / "source.json"
        self.source.parent.mkdir()
        self.source.write_bytes(b"sealed-source-receipt")

    def _run_export(self, validation: Any = None) -> Path:
        from executorch.examples.models.gemma4 import webgpu_artifact_manifest

        validator = validation if validation is not None else mock.Mock()
        with (
            mock.patch.object(
                self.export_module,
                "validate_assistant_checkpoint",
                return_value={"fixture": "assistant_checkpoint"},
            ),
            mock.patch.object(
                self.export_module,
                "build_k2_round_program",
                return_value=_FakeK2Program(),
            ),
            mock.patch.object(
                self.export_module,
                "_lower_k2_round",
                return_value=(
                    _FakeExecutorchProgram(),
                    {"fixture": "lowering"},
                ),
            ),
            mock.patch.object(
                webgpu_artifact_manifest,
                "create_mtp_manifest",
                return_value={"schema_version": 1},
            ),
            mock.patch.object(
                webgpu_artifact_manifest,
                "validate_mtp_manifest",
                side_effect=validator,
            ),
        ):
            return self.export_module.export_speculative(
                self.root / "target",
                self.root / "assistant",
                self.output,
                self.receipt,
                source_receipt_path=self.source,
            )

    def test_external_source_receipt_is_published_without_moving_the_input(
        self,
    ) -> None:
        source_bytes = self.source.read_bytes()

        self.assertEqual(self.receipt, self._run_export())

        self.assertEqual(source_bytes, self.source.read_bytes())
        self.assertEqual(
            source_bytes, (self.output_root / self.source.name).read_bytes()
        )
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.receipt.is_file())

    def test_real_export_path_delegates_to_atomic_finalizer(self) -> None:
        finalizer = mock.Mock(return_value=self.receipt)
        with mock.patch.object(
            self.export_module,
            "finalize_mtp_export",
            finalizer,
        ):
            result = self._run_export()

        self.assertEqual(result, self.receipt)
        finalizer.assert_called_once()
        (
            staging,
            output,
            receipt,
            staged_pte,
            staged_ptds,
            source_receipt,
            evidence,
        ) = finalizer.call_args.args
        self.assertEqual(staging, staged_pte.parent)
        self.assertEqual(output, self.output)
        self.assertEqual(receipt, self.receipt)
        self.assertEqual(staged_pte.name, self.output.name)
        self.assertEqual(
            [path.name for path in staged_ptds],
            [f"constants_{index}.ptd" for index in range(3)],
        )
        self.assertEqual(source_receipt, self.source)
        self.assertEqual(
            {
                "assistant_checkpoint": {"fixture": "assistant_checkpoint"},
                "k2_abi": {"fixture": "k2_abi"},
                "lowering": {"fixture": "lowering"},
                "qat_selection": {"fixture": "qat_selection"},
                "target_checkpoint": {"fixture": "target_checkpoint"},
            },
            evidence,
        )

    def test_source_basename_alias_is_rejected_before_publication(self) -> None:
        self.source = self.source.with_name(self.output.name)
        self.source.write_bytes(b"alias")

        with self.assertRaisesRegex(ValueError, "duplicate normalized artifact path"):
            self._run_export()

        self.assertFalse(self.output.exists())
        self.assertFalse(self.receipt.exists())
        self.assertEqual(b"alias", self.source.read_bytes())

    def test_failed_final_validation_rolls_back_source_and_model_artifacts(
        self,
    ) -> None:
        calls = 0

        def fail_after_publication(_root: Path, _manifest: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("injected final validation failure")

        with self.assertRaisesRegex(ValueError, "injected final validation failure"):
            self._run_export(fail_after_publication)

        self.assertEqual(b"sealed-source-receipt", self.source.read_bytes())
        self.assertFalse(self.output.exists())
        self.assertFalse((self.output_root / self.source.name).exists())
        self.assertFalse(self.receipt.exists())
