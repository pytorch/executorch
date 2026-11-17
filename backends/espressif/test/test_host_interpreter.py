# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Runs XTS3 graphs on the host build of Int8Interpreter from addresses of every
alignment and checks the outputs against the bit-exact reference interpreter.
Skipped unless ESPNN_HOST_INTERPRETER points to espnn_host_interpreter; see
backends/espressif/test/host/CMakeLists.txt.

    ESPNN_HOST_INTERPRETER=cmake-out-espnn-host/espnn_host_interpreter \
        pytest backends/espressif/test/test_host_interpreter.py
"""

from __future__ import annotations

import os
import struct
import subprocess

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchao")

import torch

from executorch.backends.espressif.reference import esp_nn_reference as reference
from executorch.backends.espressif.serialization import xts3
from executorch.backends.espressif.test import (
    test_int8_lowering as lowering_cases,
    test_int8_ops as op_cases,
)

INTERPRETER = os.environ.get("ESPNN_HOST_INTERPRETER")
pytestmark = pytest.mark.skipif(
    not INTERPRETER, reason="Set ESPNN_HOST_INTERPRETER to espnn_host_interpreter"
)


@pytest.fixture(autouse=True)
def disable_grad():
    with torch.no_grad():
        yield


def run(
    tmp_path, graph: xts3.XTS3Graph, offset: int, inputs, *options, env=None
) -> tuple[list, dict]:
    blob = tmp_path / "graph.xts3"
    blob.write_bytes(xts3.serialize_graph(graph))
    input_paths, output_paths = [], []
    for i, array in enumerate(inputs):
        path = tmp_path / f"in_{i}.bin"
        path.write_bytes(array.astype(np.int8).tobytes())
        input_paths.append(str(path))
    for o in range(len(graph.outputs)):
        output_paths.append(str(tmp_path / f"out_{offset}_{o}.bin"))
    result = subprocess.run(
        [
            INTERPRETER,
            str(blob),
            str(offset),
            *options,
            *input_paths,
            "--",
            *output_paths,
        ],
        capture_output=True,
        text=True,
        env=None if env is None else {**os.environ, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (stats,) = [line for line in result.stdout.splitlines() if line.startswith("STATS")]
    outputs = [
        np.frombuffer(open(path, "rb").read(), dtype=np.int8) for path in output_paths
    ]
    return outputs, dict(field.split("=") for field in stats.split()[1:])


def se_block(version: int):
    """An XTS3 graph of the given version, random inputs and its outputs."""
    _, program = op_cases.quantize(lowering_cases.SEBlockModel(), [(1, 3, 16, 16)])
    graph = xts3.build_graph(program)
    graph.version = version
    graph = xts3.parse_graph(xts3.serialize_graph(graph))
    interpreter = reference.ReferenceInterpreter(graph)
    generator = np.random.default_rng(0)
    inputs = [
        generator.integers(-128, 128, reference.device_shape(graph.tensors[i]))
        for i in interpreter.inputs
    ]
    expected = [
        value.astype(np.int8).reshape(-1) for value in interpreter.run_outputs(inputs)
    ]
    return graph, inputs, expected


@pytest.mark.parametrize("version", [6, 7])
def test_graph_at_any_alignment(tmp_path, version):
    graph, inputs, expected = se_block(version)
    for offset in (0, 1, 2, 3, 4, 8):
        outputs, stats = run(tmp_path, graph, offset, inputs)
        for actual, wanted in zip(outputs, expected):
            np.testing.assert_array_equal(actual, wanted)
        # A misaligned graph is copied to aligned memory first.
        assert stats["blob_copied"] == ("0" if offset == 0 else "1")
        # Version 7 graphs keep their requantization parameters in place.
        assert (stats["parameters"] == "0") == (version == 7)


@pytest.mark.parametrize("version", [6, 7])
def test_constants_copied_to_psram(tmp_path, version):
    # In a build with a PSRAM stand-in the graph is copied there and the
    # runner frees the original, so every table pointer must follow the copy.
    graph, inputs, expected = se_block(version)
    outputs, stats = run(tmp_path, graph, 0, inputs, "--constants-to-psram")
    for actual, wanted in zip(outputs, expected):
        np.testing.assert_array_equal(actual, wanted)
    assert stats["blob_copied"] == stats["psram"]


def test_arena_gets_internal_ram_first(tmp_path):
    # Internal RAM holds the arena, its 16 readable bytes past the end and the
    # default 48 KiB reserve, but not the version 6 graph's requantization
    # tables too. In a build with a PSRAM stand-in, allocating the tables
    # first would push the arena out to PSRAM.
    graph, inputs, expected = se_block(6)
    _, stats = run(tmp_path, graph, 0, inputs)
    limit = int(stats["arena"]) + 16 + 48 * 1024 + 1
    outputs, stats = run(
        tmp_path, graph, 0, inputs, env={"ESPNN_HOST_INTERNAL_BYTES": str(limit)}
    )
    for actual, wanted in zip(outputs, expected):
        np.testing.assert_array_equal(actual, wanted)
    assert int(stats["parameters"]) > 0
    assert stats["arena_internal"] == "1"


def test_partial_copy_does_not_leak_into_a_bound_output(tmp_path):
    # One COPY segment writes 4 of the output's 16 elements; the rest of a
    # caller's output buffer must not keep its previous contents.
    def tensor(tensor_id, size, flags):
        return xts3.XTS3Tensor(
            tensor_id,
            (1, size),
            xts3.XTS3DType.INT8,
            flags=flags | xts3.TENSOR_FLAG_QUANTIZED,
            quant_scale=1.0,
        )

    segment = struct.pack(
        xts3.COPY_SEGMENT,
        0,
        1,
        0,
        0,
        4,
        1,
        1,
        1,
        1,
        1,
        *([1] + [0] * 5) * 2,
        0,
        0,
        0,
        0,
        0,
    )
    graph = xts3.XTS3Graph(
        tensors=[
            tensor(0, 4, xts3.TENSOR_FLAG_GRAPH_INPUT),
            tensor(1, 16, xts3.TENSOR_FLAG_GRAPH_OUTPUT),
        ],
        operations=[
            xts3.XTS3Operation(
                0,
                xts3.XTS3OpCode.COPY,
                [0],
                [1],
                struct.pack(xts3.COPY_HEADER, 1, 0, 0) + segment,
            )
        ],
        inputs=[0],
        outputs=[1],
    )
    values = np.arange(1, 5, dtype=np.int8)
    (output,), _ = run(tmp_path, graph, 0, [values], "--external-io")
    np.testing.assert_array_equal(output, np.concatenate([values, np.zeros(12)]))
