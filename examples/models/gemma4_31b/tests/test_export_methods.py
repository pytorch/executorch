# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for export.py — verifies the produced .pte exposes all 4 methods.

This test is OFF by default because end-to-end export takes ~25-35 minutes and
significant RAM/VRAM. To run it after a real export:

    EXECUTORCH_TEST_RUN_EXPORT=1 \\
    EXECUTORCH_TEST_PTE_PATH=/tmp/gemma4_31b_vision_exports/model.pte \\
    conda run -n et python -m pytest \\
      examples/models/gemma4_31b/tests/test_export_methods.py -v

If EXECUTORCH_TEST_PTE_PATH is unset, the test attempts to find a
``model.pte`` next to this file at ``/tmp/gemma4_31b_vision_exports/``.

For a cheap structural check (no real 31B export required), see
``test_export_synthetic.py`` instead.
"""

from __future__ import annotations

import os

import pytest


_RUN = os.environ.get("EXECUTORCH_TEST_RUN_EXPORT", "0") == "1"
_DEFAULT_PTE = "/tmp/gemma4_31b_vision_exports/model.pte"
_PTE_PATH = os.environ.get("EXECUTORCH_TEST_PTE_PATH", _DEFAULT_PTE)

_EXPECTED_METHODS = {
    "embed_text",
    "vision_encoder",
    "prefill",
    "decode",
}


@pytest.mark.skipif(
    not _RUN,
    reason="Set EXECUTORCH_TEST_RUN_EXPORT=1 to enable (export is slow).",
)
def test_pte_has_four_methods():
    """Load the produced .pte and assert all 4 contract methods are present."""
    if not os.path.exists(_PTE_PATH):
        pytest.skip(f"PTE not found at {_PTE_PATH}; set EXECUTORCH_TEST_PTE_PATH.")

    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(_PTE_PATH)
    methods = set(program.method_names)
    missing = _EXPECTED_METHODS - methods
    assert not missing, (
        f"Exported .pte at {_PTE_PATH} missing methods: {sorted(missing)} "
        f"(present: {sorted(methods)})"
    )
