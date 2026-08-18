# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from textwrap import dedent

import pytest

from executorch.backends.arm.scripts.docgen import generate_vgf_op_support as docgen


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("torch.ops.aten.add.Tensor", "torch.ops.aten.add.Tensor"),
        ("aten::relu", "torch.ops.aten.relu.default"),
        ("aten.mul.Tensor", "torch.ops.aten.mul.Tensor"),
        ("edge.aten.__lshift__.Scalar", "torch.ops.aten.bitwise_left_shift.Scalar"),
        (
            "executorch_exir_dialects_edge__ops_aten_masked_fill_scalar",
            "torch.ops.aten.masked_fill.Scalar",
        ),
        ("torch.ops.aten.amax", "torch.ops.aten.amax.default"),
    ],
)
def test_normalize_pytorch_op_name(raw: str, expected: str) -> None:
    assert docgen._normalize_pytorch_op_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("None", id="none-string"),
        pytest.param("operator.getitem", id="getitem"),
        pytest.param("tosa.ADD", id="foreign-dialect"),
        pytest.param(
            "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            id="quantized-decomposed",
        ),
        pytest.param("not.an.operator", id="unknown-namespace"),
        pytest.param("torch.ops.aten.add", id="missing-overload"),
    ],
)
def test_normalize_pytorch_op_name_rejects_non_canonical_inputs(raw: str) -> None:
    assert docgen._normalize_pytorch_op_name(raw) is None


def test_normalize_pytorch_op_name_reports_diagnostics() -> None:
    diagnostics: list[str] = []

    normalized = docgen._normalize_pytorch_op_name(
        "torch.aten.ops.relu.default", diagnostics=diagnostics
    )

    assert normalized == "torch.ops.aten.relu.default"
    assert diagnostics == [
        "normalised malformed namespace: torch.aten.ops.relu.default"
    ]


def test_contextual_overload_alias_is_path_specific() -> None:
    path = Path("backends/arm/test/ops/test_amax.py")

    assert (
        docgen._normalize_pytorch_op_name("torch.ops.aten.max", path=path)
        == "torch.ops.aten.max.dim"
    )
    assert docgen._normalize_pytorch_op_name("torch.ops.aten.max") is None


def test_sort_items_respects_preferred_order_and_deduplicates() -> None:
    assert docgen._sort_items(
        ["INT", "FP", "INT", "OTHER"], docgen.SUPPORT_PROFILE_ORDER
    ) == ["FP", "INT", "OTHER"]


def _write_test_module(repo_root: Path, relative_path: str, source: str) -> Path:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    return path


def test_scan_vgf_pipeline_tests_collects_fp_and_int_coverage(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_add.py",
        """
        aten_op = "torch.ops.aten.add.Tensor"

        def test_add_vgf_no_quant():
            VgfPipeline(
                object(),
                test_data,
                aten_op=aten_op,
                exir_op="executorch_exir_dialects_edge__ops_aten_add_Tensor",
                quantize=False,
            )

        def test_add_vgf_quant():
            VgfPipeline(
                object(),
                test_data,
                aten_op=aten_op,
                exir_op=[],
                quantize=True,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_vgf_pipeline_tests(tmp_path)

    assert unresolved == []
    assert diagnostics == []
    row = rows["torch.ops.aten.add.Tensor"]
    assert row.support_profiles == {"FP", "INT"}
    assert row.dtypes == {"FP32", "INT8"}
    assert row.quantization_modes == {"8x8"}
    assert row.stages == {docgen.SOURCE_ATEN, docgen.EDGE_IR}
    assert row.classifications == {docgen.DIRECT}
    assert row.tests == {
        "backends/arm/test/ops/test_add.py::test_add_vgf_no_quant",
        "backends/arm/test/ops/test_add.py::test_add_vgf_quant",
    }


def test_scan_vgf_pipeline_tests_resolves_parametrized_operator(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_parametrized.py",
        """
        import pytest

        cases = [
            pytest.param("torch.ops.aten.neg.default", id="neg"),
            pytest.param("torch.ops.aten.abs.default", id="abs"),
        ]

        @pytest.mark.parametrize("aten_op", cases)
        def test_vgf(aten_op):
            VgfPipeline(object(), test_data, aten_op=aten_op, exir_op=[], quantize=False)
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_vgf_pipeline_tests(tmp_path)

    assert unresolved == []
    assert diagnostics == []
    assert set(rows) == {
        "torch.ops.aten.abs.default",
        "torch.ops.aten.neg.default",
    }
    assert all(row.support_profiles == {"FP"} for row in rows.values())


def test_scan_vgf_pipeline_tests_infers_runtime_coverage_for_empty_ops(
    tmp_path: Path,
) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_alias_copy.py",
        """
        aten_op = "torch.ops.aten.alias_copy.default"

        def test_alias_copy_vgf_no_quant():
            VgfPipeline(object(), test_data, aten_op=[], exir_op=[], quantize=False)
        """,
    )

    rows, unresolved, _diagnostics = docgen._scan_vgf_pipeline_tests(tmp_path)

    assert unresolved == []
    row = rows["torch.ops.aten.alias_copy.default"]
    assert row.stages == {docgen.RUNTIME_ONLY}
    assert row.classifications == {docgen.INFERRED}


def test_scan_vgf_pipeline_tests_reports_unattributed_call(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/models/test_model.py",
        """
        def test_model_vgf():
            VgfPipeline(object(), test_data, aten_op=[], exir_op=[], quantize=False)
        """,
    )

    rows, unresolved, _diagnostics = docgen._scan_vgf_pipeline_tests(tmp_path)

    assert rows == {}
    assert len(unresolved) == 1
    assert unresolved[0].function == "test_model_vgf"
    assert unresolved[0].profile == "FP"
    assert unresolved[0].reason == "no statically attributable ATen or Edge operator"


def test_scan_vgf_pipeline_tests_skips_function_level_xfail(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_xfail.py",
        """
        import pytest

        @pytest.mark.xfail(reason="unsupported")
        def test_xfailed_vgf():
            VgfPipeline(
                object(),
                test_data,
                aten_op="torch.ops.aten.relu.default",
                exir_op=[],
                quantize=False,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_vgf_pipeline_tests(tmp_path)

    assert rows == {}
    assert unresolved == []
    assert diagnostics == []


def _coverage_row() -> docgen.VgfPipelineCoverage:
    return docgen.VgfPipelineCoverage(
        exported_op="torch.ops.aten.add.Tensor",
        pytorch_apis=("torch.add", "+"),
        support_profiles={"FP", "INT"},
        dtypes={"FP32", "INT8"},
        quantization_modes={"8x8"},
        tests={"backends/arm/test/ops/test_add.py::test_add_vgf"},
        stages={docgen.SOURCE_ATEN},
        classifications={docgen.DIRECT},
    )


def test_generate_markdown_public_and_debug_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _coverage_row()
    monkeypatch.setattr(
        docgen,
        "_scan_vgf_pipeline_tests",
        lambda _repo_root: ({row.exported_op: row}, [], []),
    )

    public = docgen.generate_markdown(Path("/repo"))
    debug = docgen.generate_markdown(Path("/repo"), debug=True)

    assert "# PyTorch operator support for the VGF backend" in public
    assert "Total supported PyTorch APIs: **1**." in public
    assert "`torch.add` / `+` | FP, INT | `FP32`, `INT8` | 8x8" in public
    assert "Exported operator" not in public

    assert "Total tested exported operators: **1**." in debug
    assert "Exported operator" in debug
    assert "`torch.ops.aten.add.Tensor`" in debug
    assert "test_add_vgf" in debug


def test_generate_html_escapes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    row = docgen.VgfPipelineCoverage(
        exported_op="torch.ops.aten.fake.default",
        pytorch_apis=("torch.fake<unsafe>",),
        support_profiles={"FP"},
        dtypes={"FP32"},
    )
    monkeypatch.setattr(
        docgen,
        "_scan_vgf_pipeline_tests",
        lambda _repo_root: ({row.exported_op: row}, [], []),
    )

    page = docgen.generate_html(Path("/repo"))

    assert page.startswith("<!DOCTYPE html>")
    assert "torch.fake&lt;unsafe&gt;" in page
    assert "torch.fake<unsafe>" not in page


def test_matching_evidence_accepts_stage_equivalent_alias() -> None:
    alias = "torch.ops.aten.conv2d.default"
    tested = {
        alias: docgen.VgfPipelineCoverage(
            exported_op=alias,
            pytorch_apis=("torch.nn.Conv2d",),
            support_profiles={"FP"},
            evidence_records=[
                docgen.CoverageEvidence(
                    exported_op=alias,
                    profile="FP",
                    stage=docgen.SOURCE_ATEN,
                    classification=docgen.DIRECT,
                    test="test_conv2d.py::test_conv2d_vgf",
                    asserted_op=alias,
                )
            ],
        )
    }

    records = docgen._matching_evidence(
        tested, "torch.ops.aten.convolution.default", "FP"
    )

    assert len(records) == 1
    assert records[0].classification == docgen.STAGE_EQUIVALENT
    assert records[0].asserted_op == alias


def test_run_check_reports_missing_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    op = "torch.ops.aten.add.Tensor"
    tested = {
        op: docgen.VgfPipelineCoverage(
            exported_op=op,
            pytorch_apis=("torch.add", "+"),
            support_profiles={"FP"},
            evidence_records=[
                docgen.CoverageEvidence(
                    exported_op=op,
                    profile="FP",
                    stage=docgen.SOURCE_ATEN,
                    classification=docgen.DIRECT,
                    test="test_add.py::test_add_vgf_no_quant",
                    asserted_op=op,
                )
            ],
        )
    }
    expected = {
        op: docgen.SupportedOperatorEvidence(
            exported_op=op,
            pytorch_apis=("torch.add", "+"),
            support_profiles={"FP", "INT"},
            evidence={"registry"},
        )
    }
    monkeypatch.setattr(docgen, "_validate_configuration", lambda _root: [])
    monkeypatch.setattr(
        docgen, "_scan_vgf_pipeline_tests", lambda _root: (tested, [], [])
    )
    monkeypatch.setattr(
        docgen, "_collect_backend_supported_ops", lambda _root: expected
    )

    result = docgen.run_check(Path("/repo"))
    output = capsys.readouterr().out

    assert result == 1
    assert "missing VgfPipeline coverage" in output
    assert "`torch.ops.aten.add.Tensor` | INT" in output


def test_run_check_strict_ast_fails_on_unresolved_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved = [
        docgen.UnresolvedPipelineEvidence(
            path=Path("backends/arm/test/models/test_model.py"),
            function="test_model_vgf",
            profile="FP",
            aten_expression="[]",
            exir_expression="[]",
            reason="no statically attributable ATen or Edge operator",
        )
    ]
    monkeypatch.setattr(docgen, "_validate_configuration", lambda _root: [])
    monkeypatch.setattr(
        docgen,
        "_scan_vgf_pipeline_tests",
        lambda _root: ({}, unresolved, []),
    )
    monkeypatch.setattr(docgen, "_collect_backend_supported_ops", lambda _root: {})

    assert docgen.run_check(Path("/repo"), strict_ast=False) == 0
    assert docgen.run_check(Path("/repo"), strict_ast=True) == 1


def test_main_writes_requested_markdown_and_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docgen, "generate_markdown", lambda _root, debug=False: "md\n")
    monkeypatch.setattr(docgen, "generate_html", lambda _root, debug=False: "html\n")

    result = docgen.main(
        [
            "--repo-root",
            str(tmp_path),
            "--output",
            "generated/support.md",
            "--html",
        ]
    )

    assert result == 0
    assert (tmp_path / "generated/support.md").read_text(encoding="utf-8") == "md\n"
    assert (tmp_path / "generated/support.html").read_text(encoding="utf-8") == "html\n"
