# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import struct
from unittest.mock import patch

import numpy as np
from executorch.backends.arm import arm_vela
from executorch.backends.arm.arm_vela import (
    _BLOCK_ALIGNMENT,
    _BLOCK_HEADER,
    _EXTERNAL_REFERENCE,
    _RESERVED_BYTES,
    VelaCompileResult,
    VelaExternalBlock,
)
from executorch.backends.arm.ethosu import (
    EthosUCompileSpec,
    VelaExternalBlockPlacements,
)
from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.exir._serialize.padding import aligned_size
from executorch.exir.backend.backend_details import PreprocessResult
from pytest import raises


_COMMAND_DATA = b"COP1" + b"\x00" * 12
_WEIGHT_DATA = b"weights"


def _compile_vela_blocks(
    tmp_path,
    block_placements,
) -> VelaCompileResult:
    data = {
        "cmd_data": np.frombuffer(_COMMAND_DATA, dtype=np.uint8),
        "weight_data": np.frombuffer(_WEIGHT_DATA, dtype=np.uint8),
        "scratch_shape": np.array([32], dtype=np.int64),
        "input_shape": np.empty((0, 6), dtype=np.int32),
        "output_shape": np.empty((0, 6), dtype=np.int32),
    }
    with (
        patch.object(arm_vela, "has_vela", True),
        patch.object(arm_vela, "vela", create=True),
        patch.object(
            arm_vela.np,
            "load",
            return_value=contextlib.nullcontext(data),
        ),
    ):
        return arm_vela.vela_compile(
            b"tosa",
            [],
            intermediate_path=str(tmp_path),
            block_placements=block_placements,
        )


def _parse_vela_blocks(
    binary: bytes,
) -> list[tuple[str, bytes, tuple[int, bytes]]]:
    blocks: list[tuple[str, bytes, tuple[int, bytes]]] = []
    offset = 0
    while offset < len(binary):
        encoded_name, size, external, reserved = _BLOCK_HEADER.unpack_from(
            binary, offset
        )
        name = encoded_name.rstrip(b"\x00").decode()
        payload_start = offset + _BLOCK_HEADER.size
        blocks.append(
            (name, binary[payload_start : payload_start + size], (external, reserved))
        )
        offset = payload_start + aligned_size(size, _BLOCK_ALIGNMENT)
    return blocks


def test_external_vela_blocks_rejects_invalid_placements():
    with raises(ValueError, match="Invalid external Vela placement"):
        EthosUCompileSpec(
            "ethos-u85-256",
            external_block_placements=VelaExternalBlockPlacements(cmd_data=""),
        )


def test_vela_compile_embeds_hash_key_and_writes_named_data(tmp_path):
    result = _compile_vela_blocks(tmp_path, {"cmd_data": "mem1"})

    blocks = _parse_vela_blocks(result.processed_bytes)
    expected_key = hashlib.sha256(b"mem1\0" + _COMMAND_DATA).hexdigest()
    assert [name for name, _, _ in blocks] == [
        "vela_bin_stream",
        "cmd_data",
        "weight_data",
        "scratch_size",
        "inputs",
        "outputs",
        "vela_end_stream",
    ]
    assert blocks[1][1] == expected_key.encode("ascii")
    assert blocks[1][2] == (_EXTERNAL_REFERENCE, _RESERVED_BYTES)
    assert result.external_blocks == (
        VelaExternalBlock(expected_key, _COMMAND_DATA, 16, "mem1"),
    )


def test_vela_compile_supports_any_selected_block(tmp_path):
    payloads = {
        "cmd_data": _COMMAND_DATA,
        "weight_data": _WEIGHT_DATA,
        "scratch_size": b"\x20\x00\x00\x00",
        "inputs": b"\x00\x00\x00\x00",
        "outputs": b"\x00\x00\x00\x00",
    }
    placements = {name: "mem2" for name in payloads}
    placements["cmd_data"] = "mem1"

    result = _compile_vela_blocks(tmp_path, placements)

    blocks = _parse_vela_blocks(result.processed_bytes)
    assert [name for name, _, _ in blocks] == [
        "vela_bin_stream",
        *payloads,
        "vela_end_stream",
    ]
    for name, key, reserved in blocks[1:-1]:
        expected_key = hashlib.sha256(
            placements[name].encode() + b"\0" + payloads[name]
        ).hexdigest()
        assert key == expected_key.encode("ascii")
        assert reserved == (_EXTERNAL_REFERENCE, _RESERVED_BYTES)
    assert [
        (block.payload, block.alignment, block.placement)
        for block in result.external_blocks
    ] == [(payloads[name], 16, placements[name]) for name in payloads]


def test_vela_compile_rejects_unknown_selected_block(tmp_path):
    with raises(ValueError, match="cmnd_data"):
        _compile_vela_blocks(tmp_path, {"cmnd_data": "mem1"})


def test_vela_compile_without_external_blocks_matches_existing_format(tmp_path):
    result = _compile_vela_blocks(tmp_path, {})
    expected = b""
    for name, payload in {
        "vela_bin_stream": b"",
        "cmd_data": _COMMAND_DATA,
        "weight_data": _WEIGHT_DATA,
        "scratch_size": b"\x20\x00\x00\x00",
        "inputs": b"\x00\x00\x00\x00",
        "outputs": b"\x00\x00\x00\x00",
        "vela_end_stream": b"",
    }.items():
        encoded_name = name.encode("utf8")[:15].ljust(16, b"\x00")
        padded_payload = payload + b"\x00" * (15 - (len(payload) - 1) % 16)
        expected += encoded_name + struct.pack("<iiii", len(payload), 0, 0, 0)
        expected += padded_payload

    assert result.processed_bytes == expected
    assert result.external_blocks == ()


def test_ethosu_preprocess_outputs_external_blocks_as_named_data(tmp_path):

    def compile_tosa_flatbuffer(
        _tosa_flatbuffer,
        compile_spec,
    ) -> VelaCompileResult:
        return _compile_vela_blocks(
            tmp_path,
            compile_spec.external_block_placements.to_block_placements(),
        )

    compile_spec = EthosUCompileSpec(
        "ethos-u85-256",
        external_block_placements=VelaExternalBlockPlacements(cmd_data="mem1"),
    )
    with (
        patch.object(
            TOSABackend,
            TOSABackend._preprocess.__name__,
            return_value=PreprocessResult(processed_bytes=b"tosa"),
        ),
        patch.object(
            EthosUBackend,
            EthosUBackend._compile_tosa_flatbuffer.__name__,
            side_effect=compile_tosa_flatbuffer,
        ),
    ):
        result = EthosUBackend.preprocess(None, compile_spec._to_list())

    assert result.data_store_output is not None
    external_data = result.data_store_output.external_data
    assert list(external_data) == ["mem1"]
    key, entry = next(iter(external_data["mem1"].items()))
    assert key == hashlib.sha256(b"mem1\0" + _COMMAND_DATA).hexdigest()
    assert result.data_store_output.buffers[entry.buffer_index] == _COMMAND_DATA
