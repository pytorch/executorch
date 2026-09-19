# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import torch

from executorch.backends.webgpu.test.op_tests import generate_op_tests as g
from executorch.backends.webgpu.test.op_tests.test_suite import (
    InputSpec,
    op_test_registry,
)
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


def test_export_case_separates_upper_bound_from_runtime_inputs(monkeypatch):
    suite = op_test_registry["conv1d"]
    case = next(c for c in suite.cases if c.name == "dynamic_length_10_to_7")
    export_shapes = []
    exported_dynamic_shapes = []
    real_export = torch.export.export

    def capture_export(module, inputs, **kwargs):
        export_shapes.append(tuple(inputs[0].shape))
        exported_dynamic_shapes.append(kwargs.get("dynamic_shapes"))
        return real_export(module, inputs, **kwargs)

    monkeypatch.setattr(torch.export, "export", capture_export)
    _module, runtime_inputs, prog = g.export_case(suite, case)

    assert export_shapes == [(1, 4, 10)]
    assert exported_dynamic_shapes == [case.dynamic_shapes]
    assert tuple(runtime_inputs[0].shape) == (1, 4, 7)
    assert g._has_vulkan_delegate(prog)


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


def test_binary_shader_family_case_contract():
    expected = {
        "minimum": (
            ("2d", ((37, 41), (37, 41))),
            ("3d", ((5, 7, 11), (5, 7, 11))),
            ("broadcast_3d_2d", ((2, 3, 8), (3, 1))),
        ),
        "pow": (
            ("2d", ((37, 41), (37, 41))),
            ("3d", ((5, 7, 11), (5, 7, 11))),
            ("broadcast_3d_2d", ((2, 3, 8), (3, 1))),
        ),
        "floor_divide": (
            ("2d", ((37, 41), (37, 41))),
            ("3d", ((5, 7, 11), (5, 7, 11))),
            ("broadcast_3d_2d", ((2, 3, 8), (3, 1))),
        ),
        "mul": (
            ("same", ((8, 32), (8, 32))),
            ("bcast_lastdim", ((1, 1, 7, 896), (1, 1, 7, 1))),
            ("bcast_firstdim", ((4, 4), (1, 4))),
            ("bcast_4d_mixed", ((3, 5, 7, 11), (1, 5, 1, 11))),
            ("mixedrank", ((4,), (3, 4))),
        ),
    }

    for op, cases in expected.items():
        suite = op_test_registry[op]
        actual = tuple(
            (
                case.name,
                tuple(
                    spec.shape if isinstance(spec, InputSpec) else spec
                    for spec in case.inputs
                ),
            )
            for case in suite.cases
        )
        assert actual == cases

    ranges = {
        "minimum": ((-3.0, 3.0), (-2.0, 4.0)),
        "pow": ((0.1, 3.0), (-2.0, 3.0)),
        "floor_divide": ((-8.0, 8.0), (0.5, 4.0)),
    }
    for op, expected_ranges in ranges.items():
        case = next(
            c for c in op_test_registry[op].cases if c.name == "broadcast_3d_2d"
        )
        for spec, (start, end) in zip(case.inputs, expected_ranges):
            assert isinstance(spec, InputSpec) and callable(spec.gen)
            values = spec.gen(spec.shape).flatten()
            assert torch.isclose(values[0], torch.tensor(start))
            assert torch.isclose(values[-1], torch.tensor(end))

    assert all(
        case.golden_fn is g.cases._floor_div_golden
        for case in op_test_registry["floor_divide"].cases
    )
