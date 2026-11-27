# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from ..model.config import CODEBOOKS, MASK, ReverbConfig
from .common import masked_tokens, random_backbone, ReferenceBackbone, tiny_config


@pytest.fixture(scope="module")
def models():
    model = random_backbone(tiny_config())
    return model, ReferenceBackbone(model)


def _prompt(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 97, (11,), generator=generator),
        torch.randint(0, MASK, (CODEBOOKS, 7), generator=generator),
    )


def test_prefill_matches_the_reference_cache(models) -> None:
    model, reference = models
    ids, codes = _prompt()
    with torch.no_grad():
        reference.open(ids, codes, frames=5, bias=1.0)
        cache = model.prefill(ids, codes)
    config = model.config
    assert cache.shape == (
        2,
        config.num_layers,
        config.num_kv_heads,
        18,
        config.head_dim,
    )
    keys = torch.stack([layer.keys[0] for layer in reference.cache.layers])
    values = torch.stack([layer.values[0] for layer in reference.cache.layers])
    torch.testing.assert_close(cache[0], keys)
    torch.testing.assert_close(cache[1], values)


def test_step_matches_a_reference_main_pass(models) -> None:
    model, reference = models
    ids, codes = _prompt()
    tokens = masked_tokens(9)
    with torch.no_grad():
        reference.open(ids, codes, frames=9, bias=1.0)
        lp = reference.log_probs(tokens, range(0, 2))
        confidence, prediction, held = model.step(tokens, model.prefill(ids, codes))
    torch.testing.assert_close(confidence, lp.amax(-1))
    assert torch.equal(prediction, lp.argmax(-1))
    expected_held = lp.gather(-1, tokens[:2].clamp(max=MASK - 1).unsqueeze(-1)).squeeze(
        -1
    )
    torch.testing.assert_close(held, expected_held)
    assert confidence.dtype == held.dtype == torch.float32
    assert prediction.dtype == torch.int64


def test_polish_matches_the_reference_polish_passes(models) -> None:
    model, reference = models
    ids, codes = _prompt()
    tokens = masked_tokens(9)
    tokens[:2] = torch.randint(
        0, MASK, (2, 9), generator=torch.Generator().manual_seed(4)
    )
    with torch.no_grad():
        reference.open(ids, codes, frames=9, bias=1.0)
        expected = tokens.clone()
        first = reference.log_probs(expected, range(2, 8), polish=True)
        expected[2:4] = first[:2].argmax(-1)
        second = reference.log_probs(expected, range(4, 8), polish=True)
        expected[4:] = (second + 0.5 * (second - first[2:])).argmax(-1)
        fine = model.polish(
            tokens, torch.tensor([7]), model.prefill(ids, codes), 1.0, 0.5
        )
    assert torch.equal(fine, expected[2:])


def test_polish_ignores_codebooks_two_to_seven_of_its_input(models) -> None:
    model, _ = models
    ids, codes = _prompt()
    tokens = masked_tokens(6)
    noisy = tokens.clone()
    noisy[2:] = torch.randint(0, MASK, (6, 6))
    with torch.no_grad():
        cache = model.prefill(ids, codes)
        clean = model.polish(tokens, torch.tensor([7]), cache, 1.0, 0.5)
        dirty = model.polish(noisy, torch.tensor([7]), cache, 1.0, 0.5)
    assert torch.equal(clean, dirty)


def test_plain_embedding_without_a_rank() -> None:
    model = random_backbone(tiny_config(embedding_rank=None))
    ids, codes = _prompt()
    with torch.no_grad():
        cache = model.prefill(ids, codes)
    assert cache.shape[3] == 18


def test_config_reads_the_release_layout() -> None:
    from .common import checkpoint_dict

    config = tiny_config()
    assert ReverbConfig.from_dict(checkpoint_dict(config)) == config


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(num_audio_codebook=9), "codebooks"),
        (lambda d: d.update(quantization={"bits": 4}), "quantized"),
        (lambda d: d["llm_config"]["rope_parameters"].update(rope_type="yarn"), "RoPE"),
    ],
)
def test_config_rejects_unsupported_checkpoints(change, message) -> None:
    from .common import checkpoint_dict

    data = checkpoint_dict(tiny_config())
    change(data)
    with pytest.raises(ValueError, match=message):
        ReverbConfig.from_dict(data)
