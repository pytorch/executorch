# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import math
import shutil

import pytest
import torch
from executorch.backends.arm.vgf.tosa_custom_sdpa import (
    build_custom_sdpa_payload,
    decode_payload,
    encode_payload,
)


def test_custom_sdpa_shader_payload_round_trip():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")

    payload = build_custom_sdpa_payload(
        (1, 1, 32, 128),
        (1, 512, 8, 128),
        (1, 512, 8, 128),
        query_dtype=torch.float32,
        key_dtype=torch.float32,
        value_dtype=torch.float32,
        start_pos=511,
        dropout_p=0.0,
        is_causal=True,
        scale=0.125,
    )
    decoded = decode_payload(encode_payload(payload))

    assert decoded["entry_point"] == "main"
    assert decoded["shader_language"] == "SPIR-V"
    assert base64.b64decode(decoded["shader_code"])[:4] == b"\x03\x02\x23\x07"
    assert decoded["workgroup_sizes"] == [64, 1, 1]
    assert decoded["layout"] == "BSHD"
    assert decoded["query_shape"] == [1, 1, 32, 128]
    assert decoded["key_shape"] == [1, 512, 8, 128]
    assert decoded["value_shape"] == [1, 512, 8, 128]
    assert decoded["start_pos"] == 511
    assert decoded["scale"] == 0.125
    assert decoded["input_0_type"] == "Buffer"
    assert decoded["input_1_type"] == "Buffer"
    assert decoded["input_2_type"] == "Buffer"
    assert decoded["output_0_type"] == "Buffer"
    assert decoded["input_0_binding"] == 0
    assert decoded["input_1_binding"] == 1
    assert decoded["input_2_binding"] == 2
    assert decoded["output_0_binding"] == 3


def test_custom_sdpa_shader_payload_uses_default_attention_scale():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")

    payload = build_custom_sdpa_payload(
        (1, 1, 4, 64),
        (1, 16, 2, 64),
        (1, 16, 2, 64),
        query_dtype=torch.float32,
        key_dtype=torch.float32,
        value_dtype=torch.float32,
        start_pos=15,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
    )

    assert payload["scale"] is None
    assert payload["effective_scale"] == pytest.approx(1.0 / math.sqrt(64.0))
    assert payload["workgroup_sizes"] == [4, 1, 1]


def test_custom_sdpa_shader_payload_supports_bhsd_layout():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")

    payload = build_custom_sdpa_payload(
        (1, 32, 1, 64),
        (1, 8, 16, 64),
        (1, 8, 16, 64),
        query_dtype=torch.float32,
        key_dtype=torch.float32,
        value_dtype=torch.float32,
        start_pos=15,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
        layout="BHSD",
    )

    assert payload["layout"] == "BHSD"
    assert payload["query_shape"] == [1, 32, 1, 64]
    assert payload["key_shape"] == [1, 8, 16, 64]
    assert payload["value_shape"] == [1, 8, 16, 64]
    assert payload["effective_scale"] == pytest.approx(1.0 / math.sqrt(64.0))


def test_custom_sdpa_shader_payload_rejects_non_float32_tensors():
    with pytest.raises(
        ValueError,
        match="decode SDPA shader payload only supports float32 query tensors",
    ):
        build_custom_sdpa_payload(
            (1, 1, 4, 64),
            (1, 16, 2, 64),
            (1, 16, 2, 64),
            query_dtype=torch.float16,
            key_dtype=torch.float32,
            value_dtype=torch.float32,
            start_pos=15,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )
