# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side Gemma 4 `et_vk.gemma4_sdpa` ABI and architecture contract.

This file pins the exported head geometry and the eager custom-op fence and
numerics. It does not execute WebGPU route selection. The handler's
occupancy-based QK route, masked-QK-elision predicate, `S_kv <= 4096` boundary,
and dynamic-resize flip belong to the Dawn native lane.
"""

import copy
import unittest
from typing import Tuple

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
import torch.nn.functional as F
from executorch.examples.models.gemma4 import webgpu_artifact_manifest as wam

GEMMA_HEADS = 8
GEMMA_KV_HEADS = 1
GEMMA_HEAD_DIMS = (256, 512)

NEG_INF = float("-inf")


def _bshd(
    s_q: int, s_kv: int, head_dim: int, *, batch: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damped so `scale=1.0` logits stay off the softmax saturation floor."""
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(batch, s_q, GEMMA_HEADS, head_dim, generator=generator) * 0.05
    key = torch.randn(batch, s_kv, GEMMA_KV_HEADS, head_dim, generator=generator) * 0.05
    value = torch.randn(batch, s_kv, GEMMA_KV_HEADS, head_dim, generator=generator)
    return query, key, value


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    group = query.shape[2] // key.shape[2]
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2).repeat_interleave(group, dim=1),
        value.transpose(1, 2).repeat_interleave(group, dim=1),
        attn_mask=mask,
        scale=1.0,
    ).transpose(1, 2)


def _call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    start_pos: int = 0,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    return torch.ops.et_vk.gemma4_sdpa.default(
        query, key, value, start_pos, mask, dropout_p, is_causal, scale
    )


class Gemma4SdpaExportGeometryTest(unittest.TestCase):
    def test_shipped_config_satisfies_the_architecture_validator(self) -> None:
        # The validator only sees this config behind a full checkpoint root.
        config = wam._load_json(wam._source_config_path())
        wam._validate_architecture(config, "source config")

    def test_architecture_validator_rejects_a_drifted_config(self) -> None:
        config = copy.deepcopy(dict(wam._load_json(wam._source_config_path())))
        config["text_config"]["num_attention_heads"] = GEMMA_HEADS + 1
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            wam._validate_architecture(config, "source config")

    def test_numeric_cases_use_the_exported_head_geometry(self) -> None:
        # The damped fixtures below use the exported architecture.
        fingerprint = wam.ARCHITECTURE_FINGERPRINT
        self.assertEqual(fingerprint["num_attention_heads"], GEMMA_HEADS)
        self.assertEqual(fingerprint["num_key_value_heads"], GEMMA_KV_HEADS)
        self.assertEqual(
            (fingerprint["head_dim"], fingerprint["global_head_dim"]),
            GEMMA_HEAD_DIMS,
        )


class Gemma4SdpaAbiTest(unittest.TestCase):
    def test_every_fence_clause_fails_closed(self) -> None:
        query, key, value = _bshd(1, 8, 256)
        mask = torch.zeros(1, 8)
        wrong_batch, _, _ = _bshd(1, 8, 256, batch=2)
        wide_query, _, _ = _bshd(1, 8, 512)
        grouped_key = torch.zeros(1, 8, 3, 256)
        for name, expected, kwargs in [
            ("dropout", "dropout=0", {"dropout_p": 0.1}),
            ("causal", "causal=false", {"is_causal": True}),
            ("scale", "scale=1", {"scale": 0.5}),
        ]:
            with self.subTest(clause=name):
                with self.assertRaisesRegex(ValueError, expected):
                    _call(query, key, value, mask, **kwargs)

        for name, expected, args in [
            ("rank-3 query", "BSHD", (query[0], key, value, mask)),
            ("rank-3 key", "BSHD", (query, key[0], value, mask)),
            ("rank-3 value", "BSHD", (query, key, value[0], mask)),
            (
                "key/value mismatch",
                "shapes do not match",
                (query, key, value[:, :4], mask),
            ),
            (
                "batch mismatch",
                "shapes do not match",
                (wrong_batch, key, value, mask),
            ),
            (
                "head dim mismatch",
                "grouped-query compatible",
                (wide_query, key, value, mask),
            ),
            (
                "grouped-query mismatch",
                "grouped-query compatible",
                (query, grouped_key, grouped_key, mask),
            ),
            (
                "rank-4 mask",
                r"rank-2 \[S_q, S_kv\] mask",
                (query, key, value, mask.reshape(1, 1, 1, 8)),
            ),
            (
                "transposed mask",
                r"rank-2 \[S_q, S_kv\] mask",
                (query, key, value, mask.transpose(0, 1)),
            ),
        ]:
            with self.subTest(clause=name):
                with self.assertRaisesRegex(ValueError, expected):
                    _call(*args)


class Gemma4SdpaNumericsTest(unittest.TestCase):
    def test_decode_geometry_matches_reference(self) -> None:
        for head_dim in GEMMA_HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                query, key, value = _bshd(1, 32, head_dim)
                mask = torch.zeros(1, 32)
                torch.testing.assert_close(
                    _call(query, key, value, mask),
                    _reference(query, key, value, mask),
                    atol=1e-4,
                    rtol=1e-3,
                )

    def test_negative_infinity_mask_positions_are_inert(self) -> None:
        # Every row keeps a live prefix: an all -inf row makes softmax NaN.
        live = 5
        query, key, value = _bshd(1, 32, 256)
        mask = torch.zeros(1, 32)
        mask[:, live:] = NEG_INF

        got = _call(query, key, value, mask)
        self.assertFalse(torch.isnan(got).any())
        torch.testing.assert_close(
            got,
            _reference(query, key[:, :live], value[:, :live], torch.zeros(1, live)),
            atol=1e-4,
            rtol=1e-3,
        )


class GenericSdpaFallbackTest(unittest.TestCase):
    def test_generic_op_rejects_the_gemma4_positional_abi(self) -> None:
        # `et_vk.sdpa` keeps its 5-value schema; numerics live in
        # backends/webgpu/test/ops/test_et_vk_sdpa.py.
        query, key, value = _bshd(1, 8, 256)
        mask = torch.zeros(1, 8)
        with self.assertRaises(RuntimeError):
            torch.ops.et_vk.sdpa.default(query, key, value, 0, mask, 0.0, False, 1.0)
