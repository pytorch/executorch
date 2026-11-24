# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

pytest.importorskip("torchaudio")

from ..model.codec import CodecDecoder, CodecEncoder, ELU  # noqa: E402
from ..model.config import CODEBOOKS  # noqa: E402
from .common import sine, tiny_codec  # noqa: E402


@pytest.fixture(scope="module")
def codec():
    return tiny_codec()


def test_decoder_matches_the_codec(codec) -> None:
    codes = torch.randint(
        0, 1024, (1, CODEBOOKS, 6), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        expected = codec.decode(codes, return_dict=False)[0, 0]
        actual = CodecDecoder(codec).eval()(codes[0])
    assert actual.shape == (6 * 960,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_encoder_matches_the_codec(codec) -> None:
    wav = sine(960 * 8)
    with torch.no_grad():
        expected = codec.encode(wav[None, None], return_dict=False)[0]
        actual = CodecEncoder(codec).eval()(wav)
    assert actual.shape == (CODEBOOKS, 8)
    assert torch.equal(actual, expected)


def test_elu_matches_torch() -> None:
    x = torch.linspace(-20, 20, 1001)
    torch.testing.assert_close(ELU()(x), torch.nn.functional.elu(x))
