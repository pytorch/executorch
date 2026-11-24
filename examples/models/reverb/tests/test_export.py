# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports tiny random models through every CPU-runnable lowering path."""

import pytest
import torch

from ..export import export_reverb as er
from ..model.config import CODEBOOKS, MASK
from .common import masked_tokens, random_backbone, sine, tiny_codec, tiny_config

BOUNDS = er.ExportBounds(prompt_tokens=32, reference_frames=16, target_frames=24)


def _runtime_program(program):
    runtime = pytest.importorskip("executorch.runtime")
    return runtime.Runtime.get().load_program(program.buffer)


@pytest.fixture(scope="module")
def backbone():
    return random_backbone(tiny_config())


def test_backbone_exports_three_dynamic_methods(backbone) -> None:
    programs = er.export_backbone(backbone, BOUNDS)
    assert set(programs) == {"prefill", "step", "polish"}
    ids, codes = torch.arange(13), torch.randint(0, MASK, (CODEBOOKS, 9))
    tokens = masked_tokens(17)
    with torch.no_grad():
        cache = programs["prefill"].module()(ids, codes)
        torch.testing.assert_close(cache, backbone.prefill(ids, codes))
        step = programs["step"].module()(tokens, cache)
        expected = backbone.step(tokens, cache)
    for actual, wanted in zip(step, expected):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("backend", ["portable", "xnnpack"])
def test_backbone_program_runs(backbone, backend) -> None:
    program = er.lower(
        er.export_backbone(backbone, BOUNDS),
        backend,
        er.backbone_metadata(backbone, BOUNDS, "fp32"),
    )
    loaded = _runtime_program(program)
    assert (
        loaded.load_method("get_max_target_frames").execute([])[0]
        == BOUNDS.target_frames
    )
    ids, codes = torch.arange(13), torch.randint(0, MASK, (CODEBOOKS, 9))
    tokens = masked_tokens(17)
    cache = loaded.load_method("prefill").execute([ids, codes])[0]
    with torch.no_grad():
        expected_cache = backbone.prefill(ids, codes)
        confidence, prediction, held = backbone.step(tokens, expected_cache)
        fine = backbone.polish(
            tokens,
            torch.tensor([9]),
            expected_cache,
            er.POLISH_BIAS,
            er.POLISH_GUIDANCE,
        )
    torch.testing.assert_close(cache, expected_cache, atol=1e-4, rtol=1e-4)
    outputs = loaded.load_method("step").execute([tokens, cache])
    torch.testing.assert_close(outputs[0], confidence, atol=1e-4, rtol=1e-4)
    assert torch.equal(outputs[1], prediction)
    polished = loaded.load_method("polish").execute([tokens, torch.tensor([9]), cache])[
        0
    ]
    assert torch.equal(polished, fine)


@pytest.mark.parametrize("backend", ["portable", "xnnpack"])
def test_codec_program_runs(backend) -> None:
    pytest.importorskip("torchaudio")
    codec = tiny_codec()
    program = er.lower(
        er.export_codec(codec, BOUNDS, torch.float32),
        backend,
        er.codec_metadata(BOUNDS),
    )
    loaded = _runtime_program(program)
    wav = sine(960 * 6)
    codes = loaded.load_method("encode").execute([wav])[0]
    with torch.no_grad():
        expected = codec.encode(wav[None, None], return_dict=False)[0]
    assert torch.equal(codes, expected)
    audio = loaded.load_method("decode").execute([codes])[0]
    with torch.no_grad():
        reference = codec.decode(codes[None], return_dict=False)[0, 0]
    torch.testing.assert_close(audio, reference, atol=1e-4, rtol=1e-3)


def test_quantization_choices_are_per_backend() -> None:
    assert "8da4w" in er.QLINEAR["xnnpack"]
    assert "4w" in er.QLINEAR["mlx"]
    assert er.QLINEAR["portable"] == ()
