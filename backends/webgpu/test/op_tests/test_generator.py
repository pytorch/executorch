# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import torch

from executorch.backends.webgpu.test.op_tests import generate_op_tests as g
from executorch.backends.webgpu.test.op_tests.test_suite import op_test_registry
from executorch.backends.webgpu.test.ops.test_logical_and import (
    LOGICAL_BINARY_CASES,
    logical_binary_gen_a,
    logical_binary_gen_b,
)


def _add_regular_case():
    suite = op_test_registry["add"]
    case = next(c for c in suite.cases if c.name == "regular_2d")
    return suite, case


def test_export_case_has_delegate():
    suite, case = _add_regular_case()
    _module, _inputs, prog = g.export_case(suite, case)
    assert g._has_vulkan_delegate(prog)
    assert len(prog.buffer) > 100


def test_generate_case_writes_artifacts(tmp_path):
    suite, case = _add_regular_case()
    # generate_case returns one entry per output; add is single-output.
    entries = g.generate_case("add", suite, case, str(tmp_path))
    assert len(entries) == 1
    entry = entries[0]
    # .pte + 2 input .bin + golden .bin all exist
    assert (tmp_path / entry["pte"]).exists()
    assert len(entry["inputs"]) == 2
    for ie in entry["inputs"]:
        p = tmp_path / ie["path"]
        assert p.exists() and p.stat().st_size == np.prod(ie["shape"]) * 4
    gp = tmp_path / entry["golden"]["path"]
    assert gp.exists()
    # golden bytes == module(*materialized inputs), recomputed from the SAME .in bins
    ins = [
        torch.from_numpy(
            np.fromfile(tmp_path / ie["path"], dtype="<f4").reshape(ie["shape"]).copy()
        )
        for ie in entry["inputs"]
    ]
    expected = suite.module_factory(**case.construct)(*ins)
    got = np.fromfile(gp, dtype="<f4").reshape(entry["golden"]["shape"])
    assert np.allclose(got, expected.detach().numpy(), atol=1e-6)
    assert entry["golden"]["output_index"] == 0


def test_generate_manifest(tmp_path):
    g.generate(str(tmp_path), ops=["add"])
    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert len(data) == len(op_test_registry["add"].cases)  # 5 same-shape add cases
    for e in data:
        assert {
            "op",
            "case",
            "pte",
            "inputs",
            "golden",
            "atol",
            "rtol",
            "required",
            "heavy",
        } <= set(e)
        # add cases are non-heavy + required (export-present, FAIL-on-absence).
        assert e["required"] is True and e["heavy"] is False
        assert (tmp_path / e["pte"]).exists()
        assert (tmp_path / e["golden"]["path"]).exists()


def test_every_case_delegates():
    # Contract: every registered case must lower to a VulkanBackend delegate. An op that
    # silently CPU-falls-back would otherwise produce a misleading golden-equals-golden pass.
    for op in ("add", "mul", "sigmoid", "rms_norm"):
        suite = op_test_registry[op]
        for case in suite.cases:
            _module, _inputs, prog = g.export_case(suite, case)
            assert g._has_vulkan_delegate(prog), f"{op}/{case.name} did not delegate"


def test_manifest_schema_roundtrip(tmp_path):
    # Contract: every manifest entry carries the full driver-consumed schema, with
    # per-case tolerances propagated and output_index defaulting to 0.
    g.generate(str(tmp_path), ops=["add"])
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert len(data) == len(op_test_registry["add"].cases)
    for e in data:
        assert {
            "op",
            "case",
            "pte",
            "inputs",
            "golden",
            "atol",
            "rtol",
            "required",
            "heavy",
        } <= set(e)
        assert e["atol"] == 1e-3 and e["rtol"] == 1e-3
        assert e["required"] is True and e["heavy"] is False
        for ie in e["inputs"]:
            assert {"path", "shape", "dtype"} <= set(ie) and ie["dtype"] == "float32"
        gd = e["golden"]
        assert {"path", "shape", "dtype", "output_index"} <= set(gd)
        assert gd["output_index"] == 0


def test_logical_binary_case_contract():
    expected_cases = (
        ("2d", (4, 8)),
        ("3d", (2, 3, 8)),
        ("sq", (16, 16)),
        ("words63", (252,)),
        ("words64", (256,)),
        ("words65", (260,)),
    )
    assert LOGICAL_BINARY_CASES == expected_cases
    assert logical_binary_gen_a((8,)).tolist() == [
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
    ]
    assert logical_binary_gen_b((8,)).tolist() == [
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
    ]

    for op in ("logical_and", "bitwise_and", "logical_or", "bitwise_or"):
        suite = op_test_registry[op]
        assert tuple((case.name, case.construct["shape"]) for case in suite.cases) == (
            expected_cases
        )
        for case in suite.cases:
            assert case.required is True
            assert case.heavy is False
            assert len(case.inputs) == 2
            assert case.inputs[0].gen is logical_binary_gen_a
            assert case.inputs[1].gen is logical_binary_gen_b
