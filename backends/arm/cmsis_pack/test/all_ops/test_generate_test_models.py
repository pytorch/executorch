# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the export-time guards in generate_test_models.py.

These cover the two guards that catch a stale installed executorch (whose
exported .pte would carry old op signatures and fail on device against the
pack's current kernel arity): the source-provenance check and the schema-arity
check. They do not export any model, so they run without a torch/vela env.

"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
ALL_OPS_DIR = Path(__file__).resolve().parent
SCRIPTS = REPO_ROOT / "backends" / "arm" / "cmsis_pack" / "scripts"
sys.path.insert(0, str(ALL_OPS_DIR))
sys.path.insert(0, str(SCRIPTS))

import generate_test_models as g  # type: ignore[import-not-found]  # noqa: E402


def test_schema_arity_reads_cortex_m_operators():
    """Positional-arg counts (excluding the out kwarg) match operators.yaml."""
    if not (REPO_ROOT / "backends" / "cortex_m" / "ops" / "operators.yaml").is_file():
        pytest.skip("cortex_m operators.yaml not available")
    # quantized_add carries the fused activation_min/max args (13 positional);
    # the conv variants carry the AoT scratch tensor. These are exactly the args
    # a stale exporter omits, so they anchor the guard.
    assert g._schema_arity(REPO_ROOT, "quantized_add") == 13
    assert g._schema_arity(REPO_ROOT, "quantized_conv2d") == 13
    assert g._schema_arity(REPO_ROOT, "quantized_depthwise_conv2d") == 14


def test_schema_arity_unknown_op_is_none():
    assert g._schema_arity(REPO_ROOT, "not_a_real_op") is None


def test_schema_arity_missing_yaml_is_none(tmp_path):
    assert g._schema_arity(tmp_path, "quantized_add") is None


def test_executorch_provenance_guard_rejects_outside_tree(tmp_path):
    """A --source-dir that does not contain the imported executorch must fail;
    the message points at the reinstall fix.
    """
    with pytest.raises(SystemExit) as exc:
        g._assert_executorch_from_source(tmp_path)
    assert "outside the source tree" in str(exc.value)
    assert "Reinstall executorch" in str(exc.value)
