# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# pyre-strict
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import importlib.resources as _resources
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import ClassVar, List

import executorch.backends.vulkan.serialization as serialization_package

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkBytes,
    VkGraph,
)
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile


# Python's json module spells the non-finite floats "Infinity" / "-Infinity" /
# "NaN"; flatc spells the infinities "inf" / "-inf" and rejects Python's
# spelling, so both directions need translating. A graph carries a non-finite
# scalar whenever the model does -- the -inf fill value of a transformer
# attention mask is the common case -- and without this the failure surfaces as
# a flatc byte offset into a temporary file rather than anything pointing at
# the graph.
#
# The rewrite runs over the serialized text rather than over the encoder's
# chunks: json only emits a float as a chunk of its own inside an object, and
# inside a list the chunk carries the delimiter with it ("[-Infinity"), so
# matching whole chunks silently missed every DoubleList.
_JSON_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_PY_NONFINITE_RE = re.compile(r"(?<![\w.])(Infinity|NaN)(?![\w.])")
_FLATC_INF_RE = re.compile(r"(?<![\w.])inf(?![\w.])")


def _rewrite_outside_strings(text: str, sub) -> str:
    """Apply ``sub`` to everything in ``text`` that is not a JSON string.

    String literals are copied through untouched, so a shader name or a string
    value that happens to read "inf" is never rewritten.
    """
    out = []
    last = 0
    for m in _JSON_STRING_RE.finditer(text):
        out.append(sub(text[last : m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(sub(text[last:]))
    return "".join(out)


def _python_json_to_flatc_json(text: str) -> str:
    """Rewrite json's ``Infinity`` tokens into the ``inf`` flatc accepts."""
    if "Infinity" not in text and "NaN" not in text:
        return text

    def replace(m: "re.Match[str]") -> str:
        if m.group(1) == "NaN":
            raise ValueError(
                "Cannot serialize a NaN float value into a Vulkan graph: "
                "flatc rejects every spelling of NaN for a value inside a "
                "union, and every float in the Vulkan schema is a member of "
                "the VkValue union."
            )
        return "inf"

    return _rewrite_outside_strings(
        text, lambda segment: _PY_NONFINITE_RE.sub(replace, segment)
    )


def _flatc_json_to_python_json(text: str) -> str:
    """Rewrite flatc's bare ``inf`` tokens so json.loads accepts them."""
    if "inf" not in text:
        return text
    return _rewrite_outside_strings(
        text, lambda segment: _FLATC_INF_RE.sub("Infinity", segment)
    )


def convert_to_flatbuffer(vk_graph: VkGraph) -> bytes:
    vk_graph_json = _python_json_to_flatc_json(
        json.dumps(vk_graph, cls=_DataclassEncoder)
    )

    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                _resources.read_binary(serialization_package, "schema.fbs")
            )
        json_path = os.path.join(d, "schema.json")
        with open(json_path, "wb") as json_file:
            json_file.write(vk_graph_json.encode("ascii"))
        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "schema.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def flatbuffer_to_vk_graph(flatbuffers: bytes) -> VkGraph:
    # Following similar (de)serialization logic on other backends:
    # https://github.com/pytorch/executorch/blob/main/backends/qualcomm/serialization/qc_schema_serialize.py#L33
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                _resources.read_binary(serialization_package, "schema.fbs")
            )

        bin_path = os.path.join(d, "schema.bin")
        with open(bin_path, "wb") as bin_file:
            bin_file.write(flatbuffers)

        _flatc_decompile(d, schema_path, bin_path, ["--raw-binary"])

        json_path = os.path.join(d, "schema.json")
        with open(json_path, "rb") as output_file:
            raw = output_file.read().decode("utf-8")
        return _json_to_dataclass(json.loads(_flatc_json_to_python_json(raw)), VkGraph)


def extract_vk_flatbuffer(data: bytes) -> bytes:
    h: VulkanDelegateHeader = VulkanDelegateHeader.from_bytes(
        data[: VulkanDelegateHeader.EXPECTED_LENGTH]
    )
    start = h.flatbuffer_offset
    end = h.flatbuffer_offset + h.flatbuffer_size
    return data[start:end]


@dataclass
class VulkanDelegateHeader:
    # Defines the byte region that each component of the header corresponds to
    MAGIC_IX: ClassVar[slice] = slice(4, 8)
    HEADER_SIZE_IX: ClassVar[slice] = slice(8, 10)
    FLATBUFFER_OFFSET_IX: ClassVar[slice] = slice(10, 14)
    FLATBUFFER_SIZE_IX: ClassVar[slice] = slice(14, 18)
    BYTES_OFFSET_IX: ClassVar[slice] = slice(18, 22)
    BYTES_SIZE_IX: ClassVar[slice] = slice(22, 30)

    # magic bytes that should be at the beginning of the header
    EXPECTED_MAGIC: ClassVar[bytes] = b"VH00"
    # The length of the header in bytes
    EXPECTED_LENGTH: ClassVar[int] = 30

    # Instance attributes, @dataclass will turn these into constructor args
    flatbuffer_offset: int
    flatbuffer_size: int
    bytes_offset: int
    bytes_size: int

    @staticmethod
    def from_bytes(data: bytes) -> "VulkanDelegateHeader":
        if len(data) > VulkanDelegateHeader.EXPECTED_LENGTH:
            raise ValueError(
                f"Expected header to be {VulkanDelegateHeader.EXPECTED_LENGTH} bytes, "
                f"but got {len(data)} bytes."
            )

        magic_b: bytes = data[VulkanDelegateHeader.MAGIC_IX]

        if magic_b != VulkanDelegateHeader.EXPECTED_MAGIC:
            raise ValueError(
                f"Expected magic bytes to be {VulkanDelegateHeader.EXPECTED_MAGIC}, "
                f"but got {magic_b}."
            )

        length: int = int.from_bytes(
            data[VulkanDelegateHeader.HEADER_SIZE_IX], byteorder="little"
        )

        if length != VulkanDelegateHeader.EXPECTED_LENGTH:
            raise ValueError(
                f"Expected header to be {VulkanDelegateHeader.EXPECTED_LENGTH} bytes, "
                f"but got {length} bytes."
            )

        flatbuffer_offset_b: bytes = data[VulkanDelegateHeader.FLATBUFFER_OFFSET_IX]
        flatbuffer_size_b: bytes = data[VulkanDelegateHeader.FLATBUFFER_SIZE_IX]
        bytes_offset_b: bytes = data[VulkanDelegateHeader.BYTES_OFFSET_IX]
        bytes_size_b: bytes = data[VulkanDelegateHeader.BYTES_SIZE_IX]

        return VulkanDelegateHeader(
            flatbuffer_offset=int.from_bytes(flatbuffer_offset_b, byteorder="little"),
            flatbuffer_size=int.from_bytes(flatbuffer_size_b, byteorder="little"),
            bytes_offset=int.from_bytes(bytes_offset_b, byteorder="little"),
            bytes_size=int.from_bytes(bytes_size_b, byteorder="little"),
        )

    def is_valid(self) -> bool:
        if self.flatbuffer_size <= 0:
            return False

        expected_offset = self.flatbuffer_offset + self.flatbuffer_size
        if self.bytes_offset < expected_offset:
            return False

        if self.bytes_size < 0:
            return False

        return True

    def to_bytes(self) -> bytes:
        if not self.is_valid():
            raise ValueError("VulkanDelegateHeader instance contains invalid values")

        data: bytes = (
            # 4 bytes of padding for magic bytes, this is so that the header magic
            # bytes is in the same position as the flatbuffer header magic bytes
            b"\x00\x00\x00\x00"
            + self.EXPECTED_MAGIC
            + self.EXPECTED_LENGTH.to_bytes(2, byteorder="little")
            + self.flatbuffer_offset.to_bytes(4, byteorder="little")
            + self.flatbuffer_size.to_bytes(4, byteorder="little")
            + self.bytes_offset.to_bytes(4, byteorder="little")
            + self.bytes_size.to_bytes(8, byteorder="little")
        )

        assert len(data) == VulkanDelegateHeader.EXPECTED_LENGTH

        return data


def padding_required(data_len: int, alignment: int = 16) -> int:
    remainder: int = data_len % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


def aligned_size(data_len: int, alignment: int = 16) -> int:
    return data_len + padding_required(data_len, alignment)


def pad_to(data: bytes, size: int) -> bytes:
    if size > len(data):
        data += b"\x00" * (size - len(data))
    return data


def serialize_constant_tensors(
    vk_graph: VkGraph,
    const_tensors: List[torch.Tensor],
    raw_bytes: bytearray,
) -> None:
    # Make sure that the graph does not have any registered constants prior to calling
    # this function.
    assert len(vk_graph.constants) == 0

    current_offset = len(raw_bytes)
    for tensor in const_tensors:
        # The tensor data is stored in the named data map
        if isinstance(tensor, tuple):
            named_key, size = tensor
            vk_graph.constants.append(
                VkBytes(
                    offset=18446744073709551615,  # UINT64_MAX to indicate named data
                    length=size,
                    named_key=named_key,
                )
            )
        elif tensor is None or (
            isinstance(tensor, torch.Tensor) and tensor.numel() == 0
        ):
            vk_graph.constants.append(VkBytes(current_offset, 0))
        elif isinstance(tensor, torch.Tensor):
            array_type = ctypes.c_char * tensor.untyped_storage().nbytes()
            array = ctypes.cast(
                tensor.untyped_storage().data_ptr(),
                ctypes.POINTER(array_type),
            ).contents

            tensor_bytes = bytes(array)
            # Pad the tensor bytes to the next 16 byte boundary
            raw_bytes += tensor_bytes
            raw_bytes += b"\x00" * padding_required(len(tensor_bytes))

            vk_graph.constants.append(VkBytes(current_offset, len(tensor_bytes)))
            current_offset += aligned_size(len(tensor_bytes))
        else:
            raise ValueError(f"Unsupported constant tensor type: {type(tensor)}")


def serialize_custom_shaders(
    vk_graph: VkGraph,
    custom_shaders: List[str],
    raw_bytes: bytearray,
) -> bytes:
    # Make sure that the graph deos not have any registered shaders prior to calling
    # this function.
    assert len(vk_graph.shaders) == 0

    if len(custom_shaders) == 0:
        return b""

    else:
        raise NotImplementedError("Serializing Custom shaders are not yet supported")


def serialize_vulkan_graph(
    vk_graph: VkGraph, const_tensors: List[torch.Tensor], custom_shaders: List[str]
) -> bytes:
    raw_bytes = bytearray()
    serialize_constant_tensors(vk_graph, const_tensors, raw_bytes)
    serialize_custom_shaders(vk_graph, custom_shaders, raw_bytes)
    raw_bytes = bytes(raw_bytes)

    flatbuffer_payload = convert_to_flatbuffer(vk_graph)

    header_len = aligned_size(VulkanDelegateHeader.EXPECTED_LENGTH)
    flatbuffer_payload_len = aligned_size(len(flatbuffer_payload))
    raw_bytes_len = aligned_size(len(raw_bytes))

    header: bytes = VulkanDelegateHeader(
        flatbuffer_offset=header_len,
        flatbuffer_size=len(flatbuffer_payload),
        bytes_offset=header_len + flatbuffer_payload_len,
        bytes_size=len(raw_bytes),
    ).to_bytes()

    return b"".join(
        [
            pad_to(header, header_len),
            pad_to(flatbuffer_payload, flatbuffer_payload_len),
            pad_to(raw_bytes, raw_bytes_len),
        ]
    )
