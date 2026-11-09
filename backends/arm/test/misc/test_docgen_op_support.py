# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path
from textwrap import dedent

import executorch.backends.arm.scripts.docgen.generate_op_support as docgen

import pytest

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    ProductSupported,
)


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
    pipeline_name = sorted(docgen.BACKEND_PIPELINE_CLASS_NAMES)[0]
    rendered_source = dedent(source).replace("BackendPipeline", pipeline_name)
    path.write_text(rendered_source, encoding="utf-8")
    return path


def test_scan_backend_pipeline_tests_uses_configured_pipeline_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        docgen, "BACKEND_PIPELINE_CLASS_NAMES", frozenset({"CustomBackendPipeline"})
    )
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_custom_backend.py",
        """
        def test_custom_backend():
            BackendPipeline(
                object(),
                test_data,
                aten_op="torch.ops.aten.relu.default",
                exir_op=[],
                quantize=False,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

    assert unresolved == []
    assert diagnostics == []
    assert rows["torch.ops.aten.relu.default"].support_profiles == {"FP"}


def test_u55_backend_configuration_and_profile() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        config = docgen._activate_backend("u55")
        assert config.pipeline_class_names == frozenset({"EthosU55PipelineINT"})
        assert docgen.BACKEND_NAME == "Ethos-U55"
        assert docgen.BACKEND_TOSA_SPEC == "TOSA-1.0+INT+int16+int4+u55"
        assert docgen.DEFAULT_OUTPUT == Path(
            "docs/source/backends/arm-ethos-u/U55_op_support.md"
        )
        assert docgen.SUPPORT_PROFILE_ORDER == ["INT"]

        stmt = ast.parse("EthosU55PipelineINT(quantize=False)").body[0]
        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)
        # U55 is integer-only; an irrelevant quantize kwarg cannot turn it FP.
        assert docgen._pipeline_profile(call) == "INT"
    finally:
        docgen._activate_backend(original)


def test_u55_infrastructure_xfail_is_not_treated_as_unsupported() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("u55")
        tree = ast.parse(
            "@common.XfailIfNoCorstone300\n" "def test_u55():\n" "    pass\n"
        )
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef)
        assert not docgen._function_is_skipped_or_xfailed(function)

        semantic_tree = ast.parse(
            "@pytest.mark.xfail(reason='unsupported')\n"
            "def test_u55():\n"
            "    pass\n"
        )
        semantic_function = semantic_tree.body[0]
        assert isinstance(semantic_function, ast.FunctionDef)
        assert docgen._function_is_skipped_or_xfailed(semantic_function)
    finally:
        docgen._activate_backend(original)


def test_u85_backend_configuration_and_profile() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        config = docgen._activate_backend("u85")
        assert config.pipeline_class_names == frozenset({"EthosU85PipelineINT"})
        assert docgen.BACKEND_NAME == "Ethos-U85"
        assert docgen.BACKEND_TOSA_SPEC == "TOSA-1.0+INT+int16+int4+cf"
        assert docgen.DEFAULT_OUTPUT == Path(
            "docs/source/backends/arm-ethos-u/U85_op_support.md"
        )
        assert docgen.SUPPORT_PROFILE_ORDER == ["INT"]

        stmt = ast.parse("EthosU85PipelineINT(quantize=False)").body[0]
        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)
        # U85 is integer-only; an irrelevant quantize kwarg cannot turn it FP.
        assert docgen._pipeline_profile(call) == "INT"
    finally:
        docgen._activate_backend(original)


def test_u85_infrastructure_xfail_is_not_treated_as_unsupported() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("u85")
        tree = ast.parse(
            "@common.XfailIfNoCorstone320\n" "def test_u85():\n" "    pass\n"
        )
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef)
        assert not docgen._function_is_skipped_or_xfailed(function)

        semantic_tree = ast.parse(
            "@pytest.mark.xfail(reason='unsupported')\n"
            "def test_u85():\n"
            "    pass\n"
        )
        semantic_function = semantic_tree.body[0]
        assert isinstance(semantic_function, ast.FunctionDef)
        assert docgen._function_is_skipped_or_xfailed(semantic_function)
    finally:
        docgen._activate_backend(original)


def test_u85_explicit_coverage_attribution() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("u85")
        coverage = docgen._active_explicit_backend_coverage()

        assert coverage[
            (
                "backends/arm/test/ops/test_div_tensor_mode.py",
                "test_div_tensor_mode_u85_INT",
            )
        ]["INT"] == {"torch.ops.aten.div.Tensor_mode"}
        assert coverage[("backends/arm/test/ops/test_silu.py", "test_silu_u85_INT")][
            "INT"
        ] == {"torch.ops.aten.silu.default"}
    finally:
        docgen._activate_backend(original)


SDPA_OP = "torch.ops.aten.scaled_dot_product_attention.default"


def test_sdpa_is_decomposed_with_public_api_alias() -> None:
    assert SDPA_OP in docgen.DECOMPOSED_OPS
    assert docgen._pytorch_api_aliases(SDPA_OP) == (
        "torch.nn.functional.scaled_dot_product_attention",
    )


@pytest.mark.parametrize(
    ("backend", "function", "profile"),
    [
        ("vgf", "test_sdpa_vgf_no_quant", "FP"),
        ("vgf", "test_sdpa_vgf_quant", "INT"),
        ("u55", "test_sdpa_u55_INT", "INT"),
        ("u85", "test_sdpa_u85_INT", "INT"),
    ],
)
def test_sdpa_explicit_backend_coverage(
    backend: str, function: str, profile: str
) -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend(backend)
        coverage = docgen._active_explicit_backend_coverage()
        assert coverage[("backends/arm/test/ops/test_sdpa.py", function)][profile] == {
            SDPA_OP
        }
    finally:
        docgen._activate_backend(original)


def test_non_vgf_backend_does_not_collect_vgf_custom_partition_ops() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("u85")
        assert docgen._collect_backend_custom_partition_ops(object()) == {}  # type: ignore[arg-type]
    finally:
        docgen._activate_backend(original)


def test_backend_registry_filter_removes_unconditional_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = "torch.ops.aten.where.self"
    kept = "torch.ops.aten.add.Tensor"
    expected = {
        rejected: docgen.SupportedOperatorEvidence(
            exported_op=rejected,
            pytorch_apis=("torch.where",),
            support_profiles={"INT"},
        ),
        kept: docgen.SupportedOperatorEvidence(
            exported_op=kept, pytorch_apis=("torch.add",), support_profiles={"INT"}
        ),
    }
    monkeypatch.setattr(
        docgen,
        "_unconditionally_unsupported_backend_ops",
        lambda _spec: {rejected},
    )

    filtered = docgen._filter_backend_supported_ops(expected, object())  # type: ignore[arg-type]

    assert set(filtered) == {kept}


def test_u55_explicit_coverage_attribution() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("u55")
        coverage = docgen._active_explicit_backend_coverage()

        assert coverage[
            (
                "backends/arm/test/ops/test_div_tensor_mode.py",
                "test_div_tensor_mode_u55_INT",
            )
        ]["INT"] == {"torch.ops.aten.div.Tensor_mode"}
        assert coverage[
            (
                "backends/arm/test/ops/test_index_select.py",
                "test_index_select_u55_INT_constant_contiguous",
            )
        ]["INT"] == {"torch.ops.aten.index_select.default"}
        assert coverage[("backends/arm/test/ops/test_silu.py", "test_silu_u55_INT")][
            "INT"
        ] == {"torch.ops.aten.silu.default"}
        assert coverage[
            (
                "backends/arm/test/ops/test_unfold_copy.py",
                "test_unfold_copy_u55_INT",
            )
        ]["INT"] == {"torch.ops.aten.unfold_copy.default"}
    finally:
        docgen._activate_backend(original)


def test_u55_exported_op_exclusions_are_backend_specific() -> None:
    exclusions = docgen.BACKEND_EXPORTED_OP_EXCLUSIONS["u55"]
    assert exclusions == frozenset(
        {
            "torch.ops.aten.masked_fill.Scalar",
            "torch.ops.aten.embedding.default",
            "torch.ops.aten.sign.default",
        }
    )

    original = docgen.ACTIVE_BACKEND_KEY
    try:
        rejected = "torch.ops.aten.masked_fill.Scalar"
        kept = "torch.ops.aten.add.Tensor"

        def make_expected():
            return {
                rejected: docgen.SupportedOperatorEvidence(
                    exported_op=rejected,
                    pytorch_apis=("torch.masked_fill",),
                    support_profiles={"INT"},
                ),
                kept: docgen.SupportedOperatorEvidence(
                    exported_op=kept,
                    pytorch_apis=("torch.add",),
                    support_profiles={"INT"},
                ),
            }

        docgen._activate_backend("u55")
        filtered_u55 = docgen._filter_backend_supported_ops(make_expected(), object())  # type: ignore[arg-type]
        assert set(filtered_u55) == {kept}

        docgen._activate_backend("vgf")
        filtered_vgf = docgen._filter_backend_supported_ops(make_expected(), object())  # type: ignore[arg-type]
        assert set(filtered_vgf) == {rejected, kept}
    finally:
        docgen._activate_backend(original)


def test_product_supported_is_registered_only_for_float_specs() -> None:
    assert ProductSupported.tosa_specs
    assert all(spec.support_float() for spec in ProductSupported.tosa_specs)


def test_pipeline_profile_uses_configured_quantize_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stmt = ast.parse("BackendPipeline(quantize=False)").body[0]
    assert isinstance(stmt, ast.Expr)
    call = stmt.value
    assert isinstance(call, ast.Call)
    assert docgen._pipeline_profile(call) == "FP"

    monkeypatch.setattr(docgen, "PIPELINE_QUANTIZE_KEYWORD", None)
    monkeypatch.setattr(docgen, "PIPELINE_DEFAULT_PROFILE", "INT")
    assert docgen._pipeline_profile(call) == "INT"


def test_scan_backend_pipeline_tests_collects_fp_and_int_coverage(
    tmp_path: Path,
) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_add.py",
        """
        aten_op = "torch.ops.aten.add.Tensor"

        def test_add_vgf_no_quant():
            BackendPipeline(
                object(),
                test_data,
                aten_op=aten_op,
                exir_op="executorch_exir_dialects_edge__ops_aten_add_Tensor",
                quantize=False,
            )

        def test_add_vgf_quant():
            BackendPipeline(
                object(),
                test_data,
                aten_op=aten_op,
                exir_op=[],
                quantize=True,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

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


def test_scan_backend_pipeline_tests_resolves_parametrized_operator(
    tmp_path: Path,
) -> None:
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
            BackendPipeline(
                object(),
                test_data,
                aten_op=aten_op,
                exir_op=[],
                quantize=False,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

    assert unresolved == []
    assert diagnostics == []
    assert set(rows) == {
        "torch.ops.aten.abs.default",
        "torch.ops.aten.neg.default",
    }
    assert all(row.support_profiles == {"FP"} for row in rows.values())


def test_scan_backend_pipeline_tests_infers_runtime_coverage_for_empty_ops(
    tmp_path: Path,
) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_alias_copy.py",
        """
        aten_op = "torch.ops.aten.alias_copy.default"

        def test_alias_copy_vgf_no_quant():
            BackendPipeline(object(), test_data, aten_op=[], exir_op=[], quantize=False)
        """,
    )

    rows, unresolved, _diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

    assert unresolved == []
    row = rows["torch.ops.aten.alias_copy.default"]
    assert row.stages == {docgen.RUNTIME_ONLY}
    assert row.classifications == {docgen.INFERRED}


def test_scan_backend_pipeline_tests_reports_unattributed_call(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/models/test_model.py",
        """
        def test_model_vgf():
            BackendPipeline(object(), test_data, aten_op=[], exir_op=[], quantize=False)
        """,
    )

    rows, unresolved, _diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

    assert rows == {}
    assert len(unresolved) == 1
    assert unresolved[0].function == "test_model_vgf"
    assert unresolved[0].profile == "FP"
    assert unresolved[0].reason == "no statically attributable ATen or Edge operator"


def test_scan_backend_pipeline_tests_skips_function_level_xfail(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_xfail.py",
        """
        import pytest

        @pytest.mark.xfail(reason="unsupported")
        def test_xfailed_vgf():
            BackendPipeline(
                object(),
                test_data,
                aten_op="torch.ops.aten.relu.default",
                exir_op=[],
                quantize=False,
            )
        """,
    )

    rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)

    assert rows == {}
    assert unresolved == []
    assert diagnostics == []


def _coverage_row() -> docgen.PipelineCoverage:
    return docgen.PipelineCoverage(
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
        "_scan_backend_pipeline_tests",
        lambda _repo_root: ({row.exported_op: row}, [], []),
    )

    public = docgen.generate_markdown(Path("/repo"))
    debug = docgen.generate_markdown(Path("/repo"), debug=True)

    assert f"# {docgen.PAGE_TITLE}" in public
    assert "Total supported PyTorch APIs: **1**." in public
    assert "`torch.add` / `+` | FP, INT | `FP32`, `INT8` | 8x8" in public
    assert "Exported operator" not in public

    assert "Total tested exported operators: **1**." in debug
    assert "Exported operator" in debug
    assert "`torch.ops.aten.add.Tensor`" in debug
    assert "test_add_vgf" in debug


def test_generate_html_escapes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    row = docgen.PipelineCoverage(
        exported_op="torch.ops.aten.fake.default",
        pytorch_apis=("torch.fake<unsafe>",),
        support_profiles={"FP"},
        dtypes={"FP32"},
    )
    monkeypatch.setattr(
        docgen,
        "_scan_backend_pipeline_tests",
        lambda _repo_root: ({row.exported_op: row}, [], []),
    )

    page = docgen.generate_html(Path("/repo"))

    assert page.startswith("<!DOCTYPE html>")
    assert "torch.fake&lt;unsafe&gt;" in page
    assert "torch.fake<unsafe>" not in page


def test_matching_evidence_accepts_stage_equivalent_alias() -> None:
    alias = "torch.ops.aten.conv2d.default"
    tested = {
        alias: docgen.PipelineCoverage(
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


def test_collect_backend_custom_partition_ops_discovers_fp_and_int_profiles() -> None:
    from executorch.backends.arm.tosa import TosaSpecification

    tosa_spec = TosaSpecification.create_from_string(docgen.BACKEND_TOSA_SPEC)
    custom_ops = docgen._collect_backend_custom_partition_ops(tosa_spec)
    canonical_by_profile = {
        profile: {
            docgen._canonical_pytorch_op_from_target(target) for target in targets
        }
        for profile, targets in custom_ops.items()
    }

    expected = "torch.ops.aten.grid_sampler_2d.default"
    assert expected in canonical_by_profile["FP"]
    assert expected in canonical_by_profile["INT"]


def test_collect_backend_supported_ops_includes_vgf_custom_partition_ops() -> None:
    repo_root = Path(__file__).resolve().parents[4]

    expected = docgen._collect_backend_supported_ops(repo_root)
    row = expected["torch.ops.aten.grid_sampler_2d.default"]

    assert row.support_profiles == {"FP", "INT"}
    assert "VgfPartitioner.register_custom_partition_op" in row.evidence


def test_run_check_reports_missing_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    op = "torch.ops.aten.add.Tensor"
    tested = {
        op: docgen.PipelineCoverage(
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
        docgen, "_scan_backend_pipeline_tests", lambda _root: (tested, [], [])
    )
    monkeypatch.setattr(
        docgen, "_collect_backend_supported_ops", lambda _root: expected
    )

    result = docgen.run_check(Path("/repo"))
    output = capsys.readouterr().out

    assert result == 1
    assert f"missing {docgen.BACKEND_PIPELINE_LABEL} coverage" in output
    assert "`torch.ops.aten.add.Tensor` | INT" in output


def test_run_check_strict_ast_fails_on_unresolved_attribution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        "_scan_backend_pipeline_tests",
        lambda _root: ({}, unresolved, []),
    )
    monkeypatch.setattr(docgen, "_collect_backend_supported_ops", lambda _root: {})

    assert docgen.run_check(Path("/repo"), strict_ast=False) == 0
    assert "Unresolved VgfPipeline attribution" not in capsys.readouterr().out

    assert docgen.run_check(Path("/repo"), strict_ast=True) == 1
    assert "Unresolved VgfPipeline attribution" in capsys.readouterr().out


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


def test_main_backend_all_writes_all_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        docgen,
        "generate_markdown",
        lambda _root, debug=False: f"{docgen.ACTIVE_BACKEND_KEY}\n",
    )

    result = docgen.main(
        [
            "--repo-root",
            str(tmp_path),
            "--backend",
            "all",
        ]
    )

    assert result == 0
    for backend, config in docgen.BACKENDS.items():
        assert (tmp_path / config.default_output).read_text(encoding="utf-8") == (
            f"{backend}\n"
        )


def test_main_backend_all_check_runs_every_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[tuple[str, bool]] = []

    def fake_run_check(_root: Path, *, strict_ast: bool = False) -> int:
        checked.append((docgen.ACTIVE_BACKEND_KEY, strict_ast))
        return 1 if docgen.ACTIVE_BACKEND_KEY == "u55" else 0

    monkeypatch.setattr(docgen, "run_check", fake_run_check)

    result = docgen.main(
        [
            "--repo-root",
            str(tmp_path),
            "--backend",
            "all",
            "--check",
            "--strict-ast",
        ]
    )

    assert result == 1
    assert checked == [(backend, True) for backend in docgen.BACKENDS]


def test_main_backend_all_rejects_custom_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        docgen.main(
            [
                "--repo-root",
                str(tmp_path),
                "--backend",
                "all",
                "--output",
                "generated/support.md",
            ]
        )

    assert exc_info.value.code == 2


ADAPTIVE_AVG_POOL1D_OP = "torch.ops.aten.adaptive_avg_pool1d.default"


def test_adaptive_avg_pool1d_is_decomposed_with_public_api_alias() -> None:
    assert ADAPTIVE_AVG_POOL1D_OP in docgen.DECOMPOSED_OPS
    assert docgen._pytorch_api_aliases(ADAPTIVE_AVG_POOL1D_OP) == (
        "torch.nn.AdaptiveAvgPool1d",
        "torch.nn.functional.adaptive_avg_pool1d",
    )


def test_linalg_vector_norm_is_decomposed_with_public_api_alias() -> None:
    op = "torch.ops.aten.linalg_vector_norm.default"

    assert op in docgen.DECOMPOSED_OPS
    assert docgen._pytorch_api_aliases(op) == ("torch.linalg.vector_norm",)


@pytest.mark.parametrize(
    ("backend", "function", "profile"),
    [
        ("vgf", "test_vector_norm_vgf_no_quant", "FP"),
        ("vgf", "test_vector_norm_vgf_quant", "INT"),
        ("u55", "test_vector_norm_u55_INT_fvp", "INT"),
        ("u85", "test_vector_norm_u85_INT_fvp", "INT"),
    ],
)
def test_linalg_vector_norm_explicit_backend_coverage(
    backend: str, function: str, profile: str
) -> None:
    op = "torch.ops.aten.linalg_vector_norm.default"
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend(backend)
        coverage = docgen._active_explicit_backend_coverage()
        assert coverage[
            (
                "backends/arm/test/ops/test_linalg_vector_norm.py",
                function,
            )
        ][profile] == {op}
    finally:
        docgen._activate_backend(original)


# FP8 VGF documentation evidence tests.


def test_docgen_recognizes_vgf_fp8_pipeline_and_unquantized_profile() -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        config = docgen._activate_backend("vgf")
        assert "VgfPipelineFP8" in config.pipeline_class_names
        for expression in ("VgfPipelineFP8()", "VgfPipelineFP8[input_t]()"):
            statement = ast.parse(expression).body[0]
            assert isinstance(statement, ast.Expr)
            assert isinstance(statement.value, ast.Call)
            assert docgen._pipeline_profile(statement.value) == "FP"
    finally:
        docgen._activate_backend(original)


def test_fp8_dtype_inference_ignores_fp32_tensor_creation() -> None:
    assert docgen._dtype_hints_from_text(
        "torch.rand(2, dtype=torch.float32).to(torch.float8_e4m3fn)"
    ) == {"FP8E4M3"}
    assert docgen._dtype_hints_from_text(
        "torch.rand(2, dtype=torch.float32).to(torch.float8_e5m2)"
    ) == {"FP8E5M2"}
    assert docgen._fp8_dtypes_from_text("torch.float8_e4m3fnuz") == set()
    assert docgen._fp8_dtypes_from_text("torch.float8_e5m2fnuz") == set()


def test_scan_vgf_fp8_cases_with_factory_calls(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_fp8_matmul.py",
        """
        test_suite_fp8 = {
            "e4m3": lambda: make_case(
                lambda: torch.rand(2, dtype=torch.float32).to(torch.float8_e4m3fn)
            ),
            "e5m2": lambda: make_case(
                lambda: torch.rand(2, dtype=torch.float32).to(torch.float8_e5m2)
            ),
        }

        @common.parametrize("test_case", test_suite_fp8)
        def test_matmul_vgf_no_quant_fp8(test_case):
            VgfPipelineFP8(
                object(), test_case(),
                aten_op="torch.ops.aten.matmul.default", exir_op=[]
            )
        """,
    )
    rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
    assert unresolved == []
    assert diagnostics == []
    row = rows["torch.ops.aten.matmul.default"]
    assert row.support_profiles == {"FP"}
    assert row.dtypes == {"FP8E4M3", "FP8E5M2"}
    assert not row.quantization_modes


def test_scan_vgf_fp8_cases_from_class_attribute(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_fp8_cat.py",
        """
        class Cat:
            test_parameters_fp8 = {
                "e4m3": lambda: (
                    torch.rand(2, dtype=torch.float32).to(torch.float8_e4m3fn),
                    "fp8e4m3",
                ),
                "e5m2": lambda: (
                    torch.rand(2, dtype=torch.float32).to(torch.float8_e5m2),
                    "fp8e5m2",
                ),
            }

        @common.parametrize("test_data", Cat.test_parameters_fp8)
        def test_cat_vgf_no_quant_fp8(test_data):
            VgfPipeline(
                object(), test_data(),
                aten_op="torch.ops.aten.cat.default", exir_op=[],
                quantize=False,
                tosa_extensions=["fp8e4m3", "fp8e5m2"],
            )
        """,
    )
    rows, unresolved, _diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
    assert unresolved == []
    row = rows["torch.ops.aten.cat.default"]
    assert row.support_profiles == {"FP"}
    assert row.dtypes == {"FP8E4M3", "FP8E5M2"}


def test_scan_vgf_fp8_cast_with_suppressed_operator_assertions(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_to_copy.py",
        """
        fp8_inputs = {
            "e4m3": lambda: torch.rand(2, dtype=torch.float32).to(torch.float8_e4m3fn),
            "e5m2": lambda: torch.rand(2, dtype=torch.float32).to(torch.float8_e5m2),
        }

        @common.parametrize("test_data", fp8_inputs)
        def test_to_vgf_no_quant_fp8_with_extension(test_data):
            VgfPipeline(
                object(), test_data(), aten_op=[], exir_op=[], quantize=False
            )
        """,
    )
    rows, unresolved, _diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
    assert unresolved == []
    row = rows["torch.ops.aten._to_copy.default"]
    assert row.support_profiles == {"FP"}
    assert row.dtypes == {"FP8E4M3", "FP8E5M2"}


def test_scan_vgf_fp8_no_false_fp32_fallback(tmp_path: Path) -> None:
    _write_test_module(
        tmp_path,
        "backends/arm/test/ops/test_fp8_unresolved_dtype.py",
        """
        def test_unknown_vgf_fp8():
            VgfPipelineFP8(
                object(), opaque_inputs(),
                aten_op="torch.ops.aten.relu.default", exir_op=[],
            )
        """,
    )
    rows, unresolved, _diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
    assert unresolved == []
    assert rows["torch.ops.aten.relu.default"].dtypes == set()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Merely enabling TOSA extensions must NOT establish tensor dtype.
        ('tosa_extensions=["fp8e4m3", "fp8e5m2"]', set()),
        ('{"tosa_extensions": ["fp8e4m3"]}', set()),
        ('{"tosa_extensions": ["fp8e5m2"]}', set()),
        ("fp8e4m3", set()),
        ("fp8e5m2", set()),
        ("case_fp8e4m3", set()),
        # Real PyTorch tensor dtypes still establish FP8 evidence.
        ("torch.float8_e4m3fn", {"FP8E4M3"}),
        ("torch.float8_e5m2", {"FP8E5M2"}),
        (
            "torch.randn(4, dtype=torch.float32).to(torch.float8_e4m3fn)",
            {"FP8E4M3"},
        ),
        (
            "torch.randn(4, dtype=torch.float32).to(torch.float8_e5m2)",
            {"FP8E5M2"},
        ),
        # Unsupported FNUZ variants and larger names are not the same dtype.
        ("torch.float8_e4m3fnuz", set()),
        ("torch.float8_e5m2fnuz", set()),
        ("custom_float8_e4m3fn_suffix", set()),
    ],
)
def test_fp8_dtype_recognizer_excludes_tosa_extension_names(
    text: str, expected: set[str]
) -> None:
    assert docgen._fp8_dtypes_from_text(text) == expected
    assert (docgen._dtype_hints_from_text(text) & {"FP8E4M3", "FP8E5M2"}) == expected


def test_vgf_fp8_extensions_without_fp8_tensors_are_not_dtype_evidence(
    tmp_path: Path,
) -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_fp8_extension_only.py",
            """
            test_cases = {
                "case_e4m3": {
                    "tensor": torch.ones(4, dtype=torch.float32),
                    "tosa_extensions": ["fp8e4m3"],
                },
                "case_e5m2": {
                    "tensor": torch.ones(4, dtype=torch.float32),
                    "tosa_extensions": ["fp8e5m2"],
                },
            }

            @common.parametrize("test_data", test_cases)
            def test_relu_vgf_no_quant_with_extensions(test_data):
                VgfPipeline(
                    object(),
                    test_data,
                    aten_op="torch.ops.aten.relu.default",
                    exir_op=[],
                    quantize=False,
                    tosa_extensions=["fp8e4m3", "fp8e5m2"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.relu.default"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP32"}
    finally:
        docgen._activate_backend(original)


# Regression tests for multi-argument pytest.param dtype inference.


def test_vgf_fp8_pytest_param_all_positional_dtypes(tmp_path: Path) -> None:
    """The dtype in pytest.param(shape, dtype) must not be discarded."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_fp8_multi_param.py",
            """
            @pytest.mark.parametrize(
                "shape,dtype",
                [
                    pytest.param((2, 2), torch.float8_e4m3fn, id="e4m3"),
                    pytest.param((2, 2), torch.float8_e5m2, id="e5m2"),
                    pytest.param((2, 2), torch.float16, id="fp16"),
                    pytest.param((2, 2), torch.bfloat16, id="bf16"),
                    pytest.param(
                        (2, 2), torch.float32,
                        marks=pytest.mark.xfail(reason="unsupported"),
                        id="xfail-fp32",
                    ),
                ],
            )
            def test_matmul_vgf_no_quant_fp8(shape, dtype):
                VgfPipelineFP8(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.matmul.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.matmul.default"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP8E4M3", "FP8E5M2", "FP16", "BF16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_pytest_param_from_helper_collects_all_args(
    tmp_path: Path,
) -> None:
    """Case factories returning pytest.param also retain later dtype args."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_fp8_helper_param.py",
            """
            def make_cases():
                return [
                    pytest.param((2, 2), torch.float8_e4m3fn),
                    pytest.param((4, 4), torch.float16),
                    pytest.param(
                        (8, 8), torch.bfloat16,
                        marks=pytest.mark.xfail(reason="unsupported"),
                    ),
                ]

            @pytest.mark.parametrize("shape,dtype", make_cases())
            def test_relu_vgf_no_quant_fp8(shape, dtype):
                VgfPipelineFP8(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.relu.default"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP8E4M3", "FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_pytest_param_extensions_are_not_dtype_evidence(
    tmp_path: Path,
) -> None:
    """A later tosa_extensions argument must not claim FP8 tensors."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_fp8_extensions_param.py",
            """
            @pytest.mark.parametrize(
                "shape,dtype,extensions",
                [
                    pytest.param(
                        (2, 2), torch.float32, ["fp8e4m3", "fp8e5m2"],
                    ),
                ],
            )
            def test_add_vgf_no_quant_fp8_extensions(shape, dtype, extensions):
                VgfPipeline(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.add.Tensor"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP32"}
    finally:
        docgen._activate_backend(original)


# Regression coverage for AACR: mixed FP8 and default-FP32 parametrization.


def test_vgf_fp8_mixed_case_table_preserves_implicit_fp32(tmp_path: Path) -> None:
    """Implicit FP32 remains when a separate case converts its input to FP8."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_mixed_fp8_fp32.py",
            """
            regular_cases = {
                "default_fp32": lambda: (torch.rand(2, 2),),
                "another_default": lambda: (torch.randn(4, 4),),
            }
            fp8_cases = {
                "e4m3": lambda: (
                    torch.rand(2, 2, dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
                "e5m2": lambda: (
                    torch.rand(4, 4, dtype=torch.float32).to(torch.float8_e5m2),
                ),
            }

            @common.parametrize("test_data", regular_cases | fp8_cases)
            def test_add_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[], quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.add.Tensor"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP32", "FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_only_case_table_does_not_invent_fp32(tmp_path: Path) -> None:
    """The FP32 source buffer of .to(float8_*) is not an FP32 test case."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_only_fp8.py",
            """
            fp8_cases = {
                "e4m3": lambda: (
                    torch.rand(2, 2, dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
                "e5m2": lambda: (
                    torch.randn(2, 2, dtype=torch.float32).to(torch.float8_e5m2),
                ),
            }

            @common.parametrize("test_data", fp8_cases)
            def test_add_vgf_no_quant_fp8(test_data):
                VgfPipelineFP8(
                    object(), test_data,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.add.Tensor"].dtypes == {"FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_multi_argument_and_xfail_cases_preserve_fp32(
    tmp_path: Path,
) -> None:
    """A later dtype arg and active FP32 case count; an xfail does not."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_fp8_multi_args_mixed.py",
            """
            @pytest.mark.parametrize(
                "shape,dtype",
                [
                    pytest.param((2, 2), torch.float8_e4m3fn),
                    pytest.param((4, 4), torch.float32),
                    pytest.param(
                        (8, 8), torch.bfloat16,
                        marks=pytest.mark.xfail(reason="unsupported"),
                    ),
                ],
            )
            def test_relu_vgf_no_quant_fp8(shape, dtype):
                VgfPipelineFP8(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP32", "FP8E4M3"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_class_held_mixed_table_retains_fp32(tmp_path: Path) -> None:
    """Class attributes and dict unions must preserve default-FP32 entries."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_fp8_class_mixed.py",
            """
            class Inputs:
                defaults = {
                    "default": lambda: (torch.zeros(2, 2),),
                }
                e4m3 = {
                    "fp8": lambda: (
                        torch.randn(2, 2, dtype=torch.float32)
                        .to(torch.float8_e4m3fn),
                    ),
                }

            @common.parametrize("test_data", Inputs.defaults | Inputs.e4m3)
            def test_max_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.max.dim", exir_op=[], quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.max.dim"].dtypes == {"FP32", "FP8E4M3"}
    finally:
        docgen._activate_backend(original)


# Regression tests: AACR review of helper-hidden implicit FP32 evidence.


def test_vgf_fp8_local_helpers_restore_implicit_fp32(tmp_path: Path) -> None:
    """Mixed cases must retain FP32 when a nested local helper creates it."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_helper_fp32_mixed.py",
            """
            def make_default(shape):
                return torch.randn(shape)

            def make_nested(shape):
                return make_default(shape)

            cases = {
                "implicit_fp32": lambda: (make_nested((2, 2)),),
                "fp16": lambda: (torch.randn(2, dtype=torch.float16),),
                "bf16": lambda: (torch.randn(2, dtype=torch.bfloat16),),
                "fp8": lambda: (
                    torch.randn(2, dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {
            "FP32",
            "FP16",
            "BF16",
            "FP8E4M3",
        }
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_local_helper_final_dtype_precedes_source_fp32(
    tmp_path: Path,
) -> None:
    """Do not claim FP32 from an FP8/FP16 helper's temporary buffer."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_helper_fp8_casts.py",
            """
            def make_e4m3():
                return torch.randn(2, dtype=torch.float32).to(torch.float8_e4m3fn)

            def make_fp16():
                return torch.randn(2, dtype=torch.float32).to(torch.float16)

            cases = {
                "fp8": lambda: (make_e4m3(),),
                "fp16": lambda: (make_fp16(),),
            }

            @common.parametrize("test_data", cases)
            def test_cat_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.cat.default", exir_op=[],
                    quantize=False,
                    tosa_extensions=["fp8e4m3"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.cat.default"].dtypes == {"FP8E4M3", "FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_recursive_local_helpers_are_cycle_safe(tmp_path: Path) -> None:
    """Mutually recursive helpers must not loop or fabricate FP32 coverage."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_docgen_recursive_helpers.py",
            """
            def make_a():
                return make_b()

            def make_b():
                return make_a()

            cases = {
                "cycle": lambda: make_a(),
                "fp16": lambda: (torch.randn(2, dtype=torch.float16),),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP16", "FP32"}
    finally:
        docgen._activate_backend(original)


# AACR Patchset 5: unresolved parameter entries must not erase FP32.


def test_vgf_fp8_unknown_imported_case_retains_fp32(tmp_path: Path) -> None:
    """Opaque imported helper plus FP16 must preserve legacy FP32 coverage."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_opaque_imported_fp32.py",
            """
            cases = {
                "imported_default": lambda: imported_make_case(),
                "fp16": lambda: (torch.randn(2, dtype=torch.float16),),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP32", "FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_opaque_case_with_local_data_alias_retains_fp32(
    tmp_path: Path,
) -> None:
    """Unknown parameter cases preserve FP32 even when passed as `data`."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_local_data_alias_mixed.py",
            """
            cases = {
                "opaque": lambda: imported_default_case(),
                "fp16": lambda: (torch.randn(2, dtype=torch.float16),),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                data = test_data()
                VgfPipeline(
                    object(), data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP32", "FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_unknown_fp32_case_with_fp8_preserves_both(tmp_path: Path) -> None:
    """An opaque case plus explicit FP8 must not erase possible FP32."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_opaque_fp8_mixed.py",
            """
            cases = {
                "opaque": lambda: fixture_make_case(),
                "e4m3": lambda: (
                    torch.randn(2, dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False, tosa_extensions=["fp8e4m3"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP32", "FP8E4M3"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_resolved_only_fp8_cases_do_not_gain_fp32(tmp_path: Path) -> None:
    """Suppress the FP32 placeholder when *all* entries resolve as FP8."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_only_resolved_fp8_cases.py",
            """
            cases = {
                "e4m3": lambda: (
                    torch.randn(2, dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
                "e5m2": lambda: (
                    torch.randn(2, dtype=torch.float32).to(torch.float8_e5m2),
                ),
            }

            @common.parametrize("test_data", cases)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False, tosa_extensions=["fp8e4m3", "fp8e5m2"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_xfailed_opaque_case_does_not_force_fp32(tmp_path: Path) -> None:
    """An excluded xfail case does not make the active table unresolved."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_fp8_with_xfailed_opaque.py",
            """
            @pytest.mark.parametrize(
                "test_data",
                [
                    pytest.param(
                        lambda: opaque_imported_factory(),
                        marks=pytest.mark.xfail(reason="unsupported"),
                    ),
                    pytest.param(
                        lambda: torch.randn(2).to(torch.float8_e4m3fn),
                    ),
                ],
            )
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False, tosa_extensions=["fp8e4m3"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP8E4M3"}
    finally:
        docgen._activate_backend(original)


def test_vgf_fp8_unknown_class_held_case_preserves_fp32(tmp_path: Path) -> None:
    """Opaque module instances in class tables also retain FP32."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_class_held_opaque_cases.py",
            """
            class Cases:
                inputs = {
                    "opaque": external_instance,
                    "bf16": lambda: (torch.randn(2, dtype=torch.bfloat16),),
                }

            @common.parametrize("test_data", Cases.inputs)
            def test_relu_vgf_no_quant(test_data):
                VgfPipeline(
                    object(), test_data,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert {"FP32", "BF16"} <= rows["torch.ops.aten.relu.default"].dtypes
    finally:
        docgen._activate_backend(original)


# AACR: only operator input positions in multi-argument pytest.param count.


def test_vgf_docgen_named_params_ignore_auxiliary_dtypes(tmp_path: Path) -> None:
    """Do not advertise masks, expected outputs, or TOSA config as input
    dtypes.
    """
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_named_param_aux.py",
            """
            @pytest.mark.parametrize(
                "input_dtype,mask_dtype,expected_dtype,extensions",
                [
                    pytest.param(
                        torch.float32, torch.bool, torch.float8_e4m3fn,
                        ["fp8e4m3"], id="fp32-data",
                    ),
                    pytest.param(
                        torch.float16, torch.bool, torch.float8_e5m2,
                        ["fp8e5m2"], id="fp16-data",
                    ),
                ],
            )
            def test_add_vgf_no_quant(
                input_dtype, mask_dtype, expected_dtype, extensions
            ):
                VgfPipeline(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[], quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.add.Tensor"].dtypes == {"FP32", "FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_multi_arg_data_mask_expected_preserves_fp32(
    tmp_path: Path,
) -> None:
    """A default FP32 input plus FP8 input must not inherit BOOL mask dtype."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_named_data_mask.py",
            """
            @pytest.mark.parametrize(
                "data,mask,expected",
                [
                    pytest.param(
                        torch.rand(2), torch.ones(2, dtype=torch.bool),
                        torch.float16,
                    ),
                    pytest.param(
                        torch.rand(2, dtype=torch.float32).to(torch.float8_e4m3fn),
                        torch.ones(2, dtype=torch.bool), torch.bfloat16,
                    ),
                ],
            )
            def test_add_vgf_no_quant(data, mask, expected):
                VgfPipeline(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                    quantize=False, tosa_extensions=["fp8e4m3"],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.add.Tensor"].dtypes == {"FP32", "FP8E4M3"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_named_params_keep_input_fp8_not_mask_or_expected(
    tmp_path: Path,
) -> None:
    """Shape/dtype mapping counts FP8, not a BOOL mask or FP16 output."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_named_fp8_param.py",
            """
            @pytest.mark.parametrize(
                ("shape", "dtype", "mask", "expected"),
                [
                    pytest.param(
                        (2, 2), torch.float8_e4m3fn,
                        torch.ones(2, dtype=torch.bool), torch.float16,
                        id="fp8-with-bool-mask",
                    ),
                    pytest.param(
                        (2, 2), torch.float8_e5m2,
                        torch.ones(2, dtype=torch.bool), torch.bfloat16,
                        id="another-fp8",
                    ),
                ],
            )
            def test_add_vgf_no_quant_fp8(shape, dtype, mask, expected):
                VgfPipelineFP8(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.add.Tensor"].dtypes == {"FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_named_params_exclude_auxiliary_only_dtypes(
    tmp_path: Path,
) -> None:
    """A table of mask/expected values cannot establish FP8 or BOOL inputs."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_named_aux_only.py",
            """
            @pytest.mark.parametrize(
                "mask_dtype,expected_dtype",
                [pytest.param(torch.bool, torch.float8_e4m3fn)],
            )
            def test_add_vgf_no_quant_fp8(mask_dtype, expected_dtype):
                VgfPipelineFP8(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.add.Tensor"].dtypes == set()
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_masked_fill_uses_primary_data_dtype(tmp_path: Path) -> None:
    """BOOL mask must not replace the implicit FP32 masked_fill data tensor."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_masked_fill.py",
            """
            class MaskedFill:
                pass

            test_modules = {
                "bool_mask": lambda: (
                    MaskedFill(),
                    (
                        torch.rand(2, 2),
                        torch.ones(2, 2, dtype=torch.bool),
                        3.14,
                    ),
                ),
                "comparison_mask": lambda: (
                    MaskedFill(),
                    (torch.randn(2, 2), torch.rand(2, 2) < 0.5, 0.0),
                ),
            }

            @common.parametrize("test_module", test_modules)
            def test_masked_fill_scalar_vgf_no_quant(test_module):
                module, inputs = test_module()
                VgfPipeline(
                    module, inputs,
                    aten_op="torch.ops.aten.masked_fill.Scalar",
                    exir_op=[], quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.masked_fill.Scalar"].dtypes == {"FP32"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_named_params_xfail_not_counted(tmp_path: Path) -> None:
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_named_xfail.py",
            """
            @pytest.mark.parametrize(
                "shape,dtype,mask",
                [
                    pytest.param(
                        (2, 2), torch.float8_e4m3fn,
                        torch.bool,
                        marks=pytest.mark.xfail(reason="unsupported"),
                    ),
                    pytest.param((2, 2), torch.float32, torch.bool),
                ],
            )
            def test_relu_vgf_no_quant(shape, dtype, mask):
                VgfPipeline(
                    object(), opaque_inputs,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP32"}
    finally:
        docgen._activate_backend(original)


# Patchset 7 AACR: explicit BOOL masks must not replace data-tensor FP32.


def test_vgf_docgen_masked_fill_actual_fixture_primary_dtype(tmp_path: Path) -> None:
    """Match the real test_masked_fill module/input tuple and aten_op=[]
    flow.
    """
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_masked_fill.py",
            """
            aten_op = "torch.ops.aten.masked_fill.Scalar"

            class MaskedFill(torch.nn.Module):
                def forward(self, x, mask, value):
                    return torch.masked_fill(x, mask, value)

            def make_fp32_masked_fill_case():
                return (
                    MaskedFill(),
                    (
                        torch.rand(1, 3, 4, 5),
                        torch.ones(1, 3, 4, 5, dtype=torch.bool),
                        7.0,
                    ),
                )

            test_modules = {
                "boolean_mask": lambda: (
                    MaskedFill(),
                    (
                        torch.rand(1, 3, 4, 5),
                        torch.rand(1, 3, 4, 5) < 0.5,
                        -1.0,
                    ),
                ),
                "full_mask": lambda: make_fp32_masked_fill_case(),
            }

            @common.parametrize("test_module", test_modules)
            def test_masked_fill_scalar_vgf_no_quant(test_module):
                module, inputs = test_module()
                pipeline = VgfPipeline(
                    module, inputs, aten_op=[], quantize=False,
                )
                pipeline.run()
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.masked_fill.Scalar"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP32"}, row.dtypes
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_masked_fill_fp16_data_not_bool_mask(tmp_path: Path) -> None:
    """Explicit FP16 data with a BOOL mask must not gain FP32 or BOOL."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_masked_fill.py",
            """
            aten_op = "torch.ops.aten.masked_fill.Scalar"
            class MaskedFill(torch.nn.Module):
                def forward(self, x, mask, value):
                    return torch.masked_fill(x, mask, value)

            def make_case():
                return (MaskedFill(), (
                    torch.rand(2, dtype=torch.float16),
                    torch.ones(2, dtype=torch.bool), 1.0,
                ))

            test_modules = {"fp16": lambda: make_case()}

            @common.parametrize("test_module", test_modules)
            def test_masked_fill_scalar_vgf_no_quant(test_module):
                module, inputs = test_module()
                VgfPipeline(module, inputs, aten_op=[], quantize=False)
            """,
        )
        rows, unresolved, _ = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert rows["torch.ops.aten.masked_fill.Scalar"].dtypes == {"FP16"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_fp8_construction_buffer_is_not_fp32(tmp_path: Path) -> None:
    """A separate FP8 operator test must not inherit its FP32 cast source."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_fp8_no_false_fp32.py",
            """
            test_data = {
                "fp8": lambda: torch.randn(2).to(torch.float8_e4m3fn),
            }

            @common.parametrize("test_data", test_data)
            def test_relu_vgf_no_quant_fp8(test_data):
                VgfPipelineFP8(
                    object(), test_data(),
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, _ = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP8E4M3"}
    finally:
        docgen._activate_backend(original)


def test_vgf_docgen_masks_in_opaque_fixture_preserve_fp32(tmp_path: Path) -> None:
    """Unknown fixture signatures still find distinct non-converted FP32."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_opaque_mask.py",
            """
            cases = {
                "default": lambda: (
                    imported_model(),
                    (torch.rand(2), torch.ones(2, dtype=torch.bool)),
                ),
            }

            @common.parametrize("test_module", cases)
            def test_add_vgf_no_quant(test_module):
                model, inputs = test_module()
                VgfPipeline(
                    model, inputs,
                    aten_op="torch.ops.aten.add.Tensor", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, _ = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert "FP32" in rows["torch.ops.aten.add.Tensor"].dtypes
    finally:
        docgen._activate_backend(original)


# AACR Patchset 9: keep the parametrization name gates consistent with the
# existing _fp_is_input_parameter() classifier.


@pytest.mark.parametrize("parameter_name", ["input", "data", "inputs", "tensor", "x"])
def test_docgen_vgf_fp8_direct_input_parameter_names(
    tmp_path: Path, parameter_name: str
) -> None:
    """Recognize FP8 dtype cases for ordinary input parameter names."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_fp8_input_name.py",
            f"""
            @pytest.mark.parametrize(
                "{parameter_name}",
                [
                    pytest.param(torch.float8_e4m3fn, id="e4m3"),
                    pytest.param(torch.float8_e5m2, id="e5m2"),
                ],
            )
            def test_relu_vgf_fp8_{parameter_name}({parameter_name}):
                VgfPipelineFP8(
                    object(), {parameter_name},
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        row = rows["torch.ops.aten.relu.default"]
        assert row.support_profiles == {"FP"}
        assert row.dtypes == {"FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)


def test_docgen_vgf_unresolved_plain_input_retains_fp32(tmp_path: Path) -> None:
    """An opaque `input` case must participate in unresolved-case filtering."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_unresolved_input_name.py",
            """
            @pytest.mark.parametrize(
                "input",
                [
                    pytest.param(imported_make_case(), id="opaque"),
                    pytest.param(torch.randn(2, dtype=torch.float16), id="fp16"),
                ],
            )
            def test_relu_vgf_no_quant(input):
                VgfPipeline(
                    object(), input,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                    quantize=False,
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP16", "FP32"}
    finally:
        docgen._activate_backend(original)


def test_docgen_vgf_fp8_input_name_excludes_auxiliary_dtype(tmp_path: Path) -> None:
    """Select data dtype and ignore a named BOOL mask and expected output."""
    original = docgen.ACTIVE_BACKEND_KEY
    try:
        docgen._activate_backend("vgf")
        _write_test_module(
            tmp_path,
            "backends/arm/test/ops/test_vgf_input_with_mask.py",
            """
            @pytest.mark.parametrize(
                "input,mask,expected",
                [
                    pytest.param(
                        torch.float8_e4m3fn, torch.bool, torch.float16,
                    ),
                    pytest.param(
                        torch.float8_e5m2, torch.bool, torch.bfloat16,
                    ),
                ],
            )
            def test_relu_vgf_fp8_with_mask(input, mask, expected):
                VgfPipelineFP8(
                    object(), input,
                    aten_op="torch.ops.aten.relu.default", exir_op=[],
                )
            """,
        )
        rows, unresolved, diagnostics = docgen._scan_backend_pipeline_tests(tmp_path)
        assert not unresolved
        assert not diagnostics
        assert rows["torch.ops.aten.relu.default"].dtypes == {"FP8E4M3", "FP8E5M2"}
    finally:
        docgen._activate_backend(original)
