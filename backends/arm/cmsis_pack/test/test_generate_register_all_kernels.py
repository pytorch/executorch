# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the CMSIS Pack RegisterAllKernels.cpp generator and its shared
operator-discovery/guard contract with the PDSC component generator.

The integration tests run the generator against the live ExecuTorch source tree
(kernels/portable, kernels/quantized, backends/cortex_m), so they exercise the
real codegen pipeline rather than a synthetic fixture.

"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "backends" / "arm" / "cmsis_pack" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_components as comp  # type: ignore[import-not-found]  # noqa: E402
import generate_register_all_kernels as gen  # type: ignore[import-not-found]  # noqa: E402
import op_guards  # type: ignore[import-not-found]  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests for the shared guard contract
# ---------------------------------------------------------------------------


def test_category_to_id():
    assert op_guards.category_to_id("Portable") == "portable"
    assert op_guards.category_to_id("Cortex-M") == "cortex_m"


def test_op_to_guard():
    assert (
        op_guards.op_to_guard("add", "Portable") == "RTE_ML_EXECUTORCH_OP_PORTABLE_ADD"
    )
    # Leading underscores are stripped so an op name and its op_*.cpp filename
    # land on the same guard.
    assert (
        op_guards.op_to_guard("_adaptive_avg_pool2d", "Portable")
        == "RTE_ML_EXECUTORCH_OP_PORTABLE_ADAPTIVE_AVG_POOL2D"
    )
    assert (
        op_guards.op_to_guard("quantized_add", "Cortex-M")
        == "RTE_ML_EXECUTORCH_OP_CORTEX_M_QUANTIZED_ADD"
    )


def test_defined_kernels_finds_definitions_and_macros():
    text = """
    Tensor& add_out(KernelRuntimeContext& ctx, const Tensor& a, Tensor& out) {
      return out;
    }
    std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_out(
        KernelRuntimeContext& ctx, const Tensor& in, Tensor& a, Tensor& b, Tensor& c) {
      return {a, b, c};
    }
    DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(acos_out, std::acos)
    // not_a_kernel_out(should be ignored, in a comment)
    """
    found = op_guards._defined_kernels(text)
    assert "add_out" in found
    assert "_native_batch_norm_legit_out" in found
    assert "acos_out" in found
    assert "not_a_kernel_out" not in found


def test_discover_and_index_synthetic(tmp_path):
    op_dir = tmp_path / "backends" / "cortex_m" / "ops"
    op_dir.mkdir(parents=True)
    (op_dir / "op_foo.cpp").write_text(
        "Tensor& foo_out(KernelRuntimeContext& ctx, const Tensor& a, Tensor& out) "
        "{ return out; }\n"
    )
    components = op_guards.discover_components(tmp_path)
    assert len(components) == 1
    component = components[0]
    assert component.category == "Cortex-M"
    assert component.guard == "RTE_ML_EXECUTORCH_OP_CORTEX_M_FOO"

    index = op_guards.kernel_source_index(tmp_path, components)
    assert index[("Cortex-M", "foo_out")] is component


def test_shared_kernel_files_bundles_non_op_helpers(tmp_path):
    """A shared kernel helper (non-op_*.cpp, outside util/pattern) must be
    bundled into the Runtime component so dependent operators link; op_*.cpp and
    util/pattern sources must not be (the per-op components and Kernel Utils own
    those).
    """
    cpu = tmp_path / "kernels" / "quantized" / "cpu"
    (cpu / "util").mkdir(parents=True)
    (cpu / "embeddingxb.cpp").write_text("// shared xbit impl\n")
    (cpu / "op_embedding2b.cpp").write_text("// op wrapper\n")
    (cpu / "util" / "refcount.cpp").write_text("// util, owned by Kernel Utils\n")

    files = comp.generate_shared_kernel_files(tmp_path)
    assert "kernels/quantized/cpu/embeddingxb.cpp" in files
    assert "op_embedding2b.cpp" not in files
    assert "refcount.cpp" not in files


# ---------------------------------------------------------------------------
# Integration tests against the live source tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repo_groups():
    if not (REPO_ROOT / "kernels" / "portable" / "functions.yaml").is_file():
        pytest.skip("ExecuTorch kernel sources not available")
    return gen.build_groups(REPO_ROOT, allow_mismatch=True)


@pytest.fixture(scope="module")
def repo_output(repo_groups):
    return gen.generate_register_all_kernels(
        repo_groups.get("Portable", []),
        repo_groups.get("Quantized", []),
        repo_groups.get("Cortex-M", []),
    )


def _registered_ops(output: str) -> set:
    return set(
        re.findall(
            r'Kernel\(\s*"((?:aten|quantized_decomposed|dim_order_ops|cortex_m)::[^"]+)"',
            output,
        )
    )


def test_known_ops_registered(repo_output):
    ops = _registered_ops(repo_output)
    assert "aten::add.out" in ops
    assert "cortex_m::quantized_add.out" in ops
    assert any(op.startswith("quantized_decomposed::") for op in ops)
    assert len(ops) > 200


def test_every_registration_is_guarded(repo_output):
    """Every Kernel registration must sit inside an #ifdef guard block."""
    depth = 0
    for line in repo_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("#ifdef RTE_ML_EXECUTORCH_OP_"):
            depth += 1
        elif stripped.startswith("#endif"):
            depth = max(0, depth - 1)
        elif stripped.startswith('Kernel("'):
            assert depth > 0, f"unguarded registration: {stripped}"


def test_cortex_m_calls_use_cortex_m_native(repo_output):
    # quantized_batch_matmul is a Cortex-M-only kernel (no quantized_decomposed
    # twin), so it unambiguously checks the namespace it is emitted under.
    assert "cortex_m::native::quantized_batch_matmul_out(context" in repo_output
    assert "torch::executor::native::quantized_batch_matmul_out" not in repo_output


def test_repo_is_reconciliation_clean():
    """The live tree must have a registration #ifdef for every component and
    vice versa -- build_groups raises (strict) on any mismatch.
    """
    if not (REPO_ROOT / "kernels" / "portable" / "functions.yaml").is_file():
        pytest.skip("ExecuTorch kernel sources not available")
    gen.build_groups(REPO_ROOT, allow_mismatch=False)


def test_cross_script_guard_sets_match(repo_groups):
    """The registration #ifdef set must equal the component #define set."""
    emitted = {g.guard for groups in repo_groups.values() for g in groups}
    result = comp.generate_all_components(REPO_ROOT, {}, "1.0.0")
    component_guards = {
        op.guard
        for key in ("portable_ops", "quantized_ops", "cortex_m_ops")
        for op in result[key]
    }
    assert emitted == component_guards


# ---------------------------------------------------------------------------
# Reconciliation / no-silent-drop
# ---------------------------------------------------------------------------


def test_reconcile_raises_on_mismatch():
    with pytest.raises(SystemExit):
        gen._reconcile(
            unmatched_yaml=[],
            unmatched_source=["RTE_ML_EXECUTORCH_OP_PORTABLE_GHOST"],
            allow_mismatch=False,
        )


def test_reconcile_allow_mismatch_warns(capsys):
    gen._reconcile(
        unmatched_yaml=[("Portable", "ghost.out", "ghost_out")],
        unmatched_source=[],
        allow_mismatch=True,
    )
    assert "WARNING" in capsys.readouterr().err


def test_component_without_registration_is_not_silent(tmp_path):
    """A staged op_*.cpp with no YAML registration must fail loudly, not be
    silently dropped.
    """
    op_dir = tmp_path / "backends" / "cortex_m" / "ops"
    op_dir.mkdir(parents=True)
    (op_dir / "op_ghost.cpp").write_text(
        "Tensor& ghost_out(KernelRuntimeContext& ctx, const Tensor& a, Tensor& out) "
        "{ return out; }\n"
    )
    with pytest.raises(SystemExit, match="GHOST"):
        gen.build_groups(tmp_path, allow_mismatch=False)
