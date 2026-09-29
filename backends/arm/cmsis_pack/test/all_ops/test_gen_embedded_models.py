# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the embedded-models generator.

Verify that gen_embedded_models.build() emits a well-formed stub when no
manifest is present (build/link coverage), and correct .incbin / table sources
with project-relative paths when models are present (so the same generated file
resolves on the host and inside the Docker build container).

"""

import json
import sys
from pathlib import Path

import pytest

ALL_OPS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALL_OPS_DIR))

import gen_embedded_models  # type: ignore[import-not-found]  # noqa: E402


def test_stub_when_no_manifest(tmp_path):
    blob_s, cpp, header, n_models, n_bytes = gen_embedded_models.build(
        tmp_path / "models", tmp_path
    )
    assert n_models == 0 and n_bytes == 0
    assert ".incbin" not in blob_s
    assert "g_embedded_models_count = 0" in cpp
    # Well-formed: a 1-element dummy array (C++ has no zero-length arrays).
    assert "g_embedded_models[]" in cpp
    assert "struct EmbeddedModel" in header


def _make_models(root: Path) -> Path:
    """Two self-contained .pte files (test data embedded in the pte)."""
    models = root / "models"
    models.mkdir(parents=True)
    (models / "portable__abs.pte").write_bytes(b"\x01\x02\x03\x04")
    (models / "cortex_m__quantized_add.pte").write_bytes(b"\x05\x06")
    manifest = [
        {"op": "abs", "category": "Portable", "name": "portable__abs"},
        {
            "op": "quantized_add",
            "category": "Cortex-M",
            "name": "cortex_m__quantized_add",
        },
    ]
    (models / "manifest.json").write_text(json.dumps(manifest))
    return models


def test_real_models_emit_relative_incbin(tmp_path):
    models = _make_models(tmp_path)
    blob_s, cpp, header, n_models, n_bytes = gen_embedded_models.build(models, tmp_path)

    assert n_models == 2
    assert n_bytes == 4 + 2
    # Paths are project-relative (resolve via -I<project-dir>), never absolute.
    assert '.incbin "models/portable__abs.pte"' in blob_s
    assert '.incbin "models/cortex_m__quantized_add.pte"' in blob_s
    assert "/tmp" not in blob_s and str(tmp_path) not in blob_s  # nosec B108
    # Table carries op name + category and both ops.
    assert '"abs", "Portable"' in cpp
    assert '"quantized_add", "Cortex-M"' in cpp
    assert cpp.count("g_embedded_models[]") == 1


def test_missing_pte_fails_generation(tmp_path):
    """A model listed in the manifest but absent on disk must abort the build:
    dropping it would shrink the embedded set and the firmware would report a
    full pass over fewer ops.
    """
    models = _make_models(tmp_path)
    (models / "portable__abs.pte").unlink()
    with pytest.raises(SystemExit) as exc:
        gen_embedded_models.build(models, tmp_path)
    assert "portable__abs.pte" in str(exc.value)


def test_count_is_sizeof_based(tmp_path):
    _, cpp, _, _, _ = gen_embedded_models.build(_make_models(tmp_path), tmp_path)
    assert (
        "g_embedded_models_count = sizeof(g_embedded_models) / "
        "sizeof(g_embedded_models[0])" in cpp
    )
