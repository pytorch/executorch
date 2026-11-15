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


def run(tmp_path, graph: xts3.XTS3Graph, offset: int, inputs) -> tuple[list, dict]:
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
        [INTERPRETER, str(blob), str(offset), *input_paths, "--", *output_paths],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (stats,) = [line for line in result.stdout.splitlines() if line.startswith("STATS")]
    outputs = [
        np.frombuffer(open(path, "rb").read(), dtype=np.int8) for path in output_paths
    ]
    return outputs, dict(field.split("=") for field in stats.split()[1:])


@pytest.mark.parametrize("version", [6, 7])
def test_graph_at_any_alignment(tmp_path, version):
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
    for offset in (0, 1, 2, 3, 4, 8):
        outputs, stats = run(tmp_path, graph, offset, inputs)
        for actual, wanted in zip(outputs, expected):
            np.testing.assert_array_equal(actual, wanted)
        # A misaligned graph is copied to aligned memory first.
        assert stats["blob_copied"] == ("0" if offset == 0 else "1")
        # Version 7 graphs keep their requantization parameters in place.
        assert (stats["parameters"] == "0") == (version == 7)
