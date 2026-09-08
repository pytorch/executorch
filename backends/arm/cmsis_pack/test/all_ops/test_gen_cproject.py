# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the all-ops consumer-project generator.

These verify that the generated all_ops.cproject.yml selects exactly the
operator components the pack ships (as enumerated by op_guards), so the consumer
build covers the whole operator surface and never drifts from the pack's PDSC.

"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
ALL_OPS_DIR = Path(__file__).resolve().parent
SCRIPTS = REPO_ROOT / "backends" / "arm" / "cmsis_pack" / "scripts"
sys.path.insert(0, str(ALL_OPS_DIR))
sys.path.insert(0, str(SCRIPTS))

import gen_cproject  # type: ignore[import-not-found]  # noqa: E402
import op_guards  # type: ignore[import-not-found]  # noqa: E402


@pytest.fixture(scope="module")
def cproject_text():
    if not (REPO_ROOT / "kernels" / "portable" / "cpu").is_dir():
        pytest.skip("ExecuTorch kernel sources not available")
    return gen_cproject.build_cproject(REPO_ROOT)


def _selected_op_components(text: str) -> set:
    prefix = "    - component: Machine Learning:ExecuTorch Operators:"
    return {
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    }


def test_base_components_present(cproject_text):
    for component in gen_cproject._BASE_COMPONENTS:
        assert f"    - component: {component}" in cproject_text


def test_selects_every_op_component(cproject_text):
    """Every component op_guards discovers must be selected, and nothing
    extra.
    """
    discovered = {
        f"{c.category} {c.name}" for c in op_guards.discover_components(REPO_ROOT)
    }
    assert _selected_op_components(cproject_text) == discovered
    assert len(discovered) > 150  # sanity: the full portable+quantized+cortex_m surface


def test_cmsis_nn_added_when_cortex_m_present(cproject_text):
    components = op_guards.discover_components(REPO_ROOT)
    has_cortex_m = any(c.category == "Cortex-M" for c in components)
    assert has_cortex_m  # repo ships Cortex-M ops
    assert f"    - component: {gen_cproject._CMSIS_NN_COMPONENT}" in cproject_text


def test_app_group_references_main(cproject_text):
    assert "- file: ./main.cpp" in cproject_text


def test_targets_cortex_m85_linker(cproject_text):
    """The image links against the SSE-315/320 (Cortex-M85) DDR layout."""
    assert "./SSE315_large.ld" in cproject_text
    assert "./SSE315_large.sct" in cproject_text
    assert "cortex-m85" in cproject_text
    assert "cortex-m55" not in cproject_text


def test_cpu_variant_omits_ethos_u(cproject_text):
    for component in gen_cproject._ETHOS_U_COMPONENTS:
        assert f"    - component: {component}" not in cproject_text


def test_ethos_u_variant_adds_backend_and_driver():
    if not (REPO_ROOT / "kernels" / "portable" / "cpu").is_dir():
        pytest.skip("ExecuTorch kernel sources not available")
    text = gen_cproject.build_cproject(REPO_ROOT, ethos_u=True)
    for component in gen_cproject._ETHOS_U_COMPONENTS:
        assert f"    - component: {component}" in text
    # Operator selection is identical to the CPU variant: every op still links.
    assert _selected_op_components(text) == _selected_op_components(
        gen_cproject.build_cproject(REPO_ROOT)
    )


def test_empty_source_dir_errors(tmp_path):
    with pytest.raises(SystemExit):
        gen_cproject.build_cproject(tmp_path)
