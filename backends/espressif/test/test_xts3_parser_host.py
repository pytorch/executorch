# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Builds the device-side XTS3 parser (espnn_runtime/src/xts3_graph.cpp, which
has no ESP-IDF dependencies) for the host and checks that it accepts graphs
produced by the serializer and rejects corrupted ones: overflowing COPY
extents and segment counts, oversized UPSAMPLE tables and misaligned int32
constants. Skipped when no C++ compiler is available.

    pytest backends/espressif/test/test_xts3_parser_host.py
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torchao")

from executorch.backends.espressif.serialization import xts3  # noqa: E402
from executorch.backends.espressif.test import test_int8_ops as ops  # noqa: E402

RUNTIME = Path(__file__).resolve().parents[1] / "espnn_runtime"
DRIVER = Path(__file__).resolve().parent / "host" / "xts3_parse_main.cpp"
INT32_MAX = 2**31 - 1


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path_factory.mktemp("xts3") / "xts3_parse"
    sources = [str(RUNTIME / "src" / "xts3_graph.cpp"), str(DRIVER)]
    base = [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror"]
    base += ["-I", str(RUNTIME / "include"), *sources, "-o", str(binary)]
    sanitized = base[:1] + ["-fsanitize=address,undefined"] + base[1:]
    for command in (sanitized, base):
        if subprocess.run(command, capture_output=True).returncode == 0:
            break
    else:
        result = subprocess.run(base, capture_output=True, text=True)
        pytest.fail(f"host build of xts3_graph.cpp failed:\n{result.stderr}")

    def run(blobs: dict[str, bytes]) -> dict[str, str]:
        folder = binary.parent
        paths = []
        for name, blob in blobs.items():
            path = folder / f"{name}.xts3"
            path.write_bytes(blob)
            paths.append(str(path))
        result = subprocess.run(
            [str(binary), *paths],
            capture_output=True,
            text=True,
            env={**os.environ, "UBSAN_OPTIONS": "halt_on_error=1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        verdicts = {}
        for line in result.stdout.splitlines():
            if line.startswith("VERDICT "):
                path, verdict = line[len("VERDICT ") :].rsplit(" ", 1)
                verdicts[Path(path).stem] = verdict
        return verdicts

    return run


def _lowered(model, shapes) -> xts3.XTS3Graph:
    _, program = ops.quantize(model, shapes)
    return xts3.build_graph(program)


def _layout(graph: xts3.XTS3Graph):
    io_table = 0
    if graph.version >= 6:
        io_table = xts3.IO_TABLE_HEADER_SIZE + 4 * (
            len(graph.inputs) + len(graph.outputs)
        )
    tensors = xts3.HEADER_SIZE + io_table
    operations = tensors + xts3.TENSOR_SIZE * len(graph.tensors)
    parameters = operations + xts3.OP_SIZE * len(graph.operations)
    offsets, cursor = [], parameters
    for operation in graph.operations:
        offsets.append(cursor)
        cursor += len(operation.params)
    return tensors, offsets


def _first(graph, opcode) -> int:
    return next(i for i, op in enumerate(graph.operations) if op.opcode == opcode)


def _patch(blob: bytes, offset: int, fmt: str, *values) -> bytes:
    data = bytearray(blob)
    struct.pack_into(fmt, data, offset, *values)
    return bytes(data)


def test_parser_accepts_serialized_graphs(parser):
    blobs = {
        name: xts3.serialize_graph(_lowered(model, shapes))
        for name, model, shapes, _ in ops.CASES
    }
    assert set(parser(blobs).values()) == {"accept"}


def test_parser_rejects_copy_extent_overflow(parser):
    graph = _lowered(ops.TransposeSelect(), [(1, 3, 4, 5)])
    blob = xts3.serialize_graph(graph)
    _, offsets = _layout(graph)
    segment = offsets[_first(graph, xts3.XTS3OpCode.COPY)] + xts3.COPY_HEADER_SIZE
    big = [INT32_MAX] * 3 + [1] * 3
    corrupted = _patch(blob, segment + 4, "<i", 3)
    corrupted = _patch(corrupted, segment + 16, "<6i", *big)
    corrupted = _patch(corrupted, segment + 40, "<6i", *big)
    corrupted = _patch(corrupted, segment + 64, "<6i", *big)
    assert parser({"valid": blob, "extent": corrupted}) == {
        "valid": "accept",
        "extent": "reject",
    }


def test_parser_rejects_copy_segment_count_overflow(parser):
    graph = _lowered(ops.TransposeSelect(), [(1, 3, 4, 5)])
    blob = xts3.serialize_graph(graph)
    _, offsets = _layout(graph)
    index = _first(graph, xts3.XTS3OpCode.COPY)
    size = len(graph.operations[index].params)
    wrapped = (2**32 + size - xts3.COPY_HEADER_SIZE) // xts3.COPY_SEGMENT_SIZE
    corrupted = _patch(blob, offsets[index], "<i", wrapped)
    assert parser({"segments": corrupted}) == {"segments": "reject"}


def test_parser_rejects_oversized_upsample_tables(parser):
    graph = _lowered(ops.UpsampleModel("nearest"), [(1, 3, 16, 16)])
    blob = xts3.serialize_graph(graph)
    _, offsets = _layout(graph)
    header = offsets[_first(graph, xts3.XTS3OpCode.UPSAMPLE)]
    corrupted = _patch(blob, header + 12, "<2i", 360_000_000, 360_000_000)
    assert parser({"upsample": corrupted}) == {"upsample": "reject"}


def test_parser_rejects_misaligned_int32_constants(parser):
    graph = _lowered(ops.DetectionHead(), [(1, 3, 16, 16)])
    blob = xts3.serialize_graph(graph)
    tensors, _ = _layout(graph)
    index = next(
        i
        for i, tensor in enumerate(graph.tensors)
        if tensor.is_constant and tensor.dtype == xts3.XTS3DType.INT32
    )
    record = tensors + xts3.TENSOR_SIZE * index
    offset = graph.tensors[index].data_offset
    corrupted = _patch(blob, record + 4, "<I", offset + 2)
    assert parser({"aligned": blob, "misaligned": corrupted}) == {
        "aligned": "accept",
        "misaligned": "reject",
    }
