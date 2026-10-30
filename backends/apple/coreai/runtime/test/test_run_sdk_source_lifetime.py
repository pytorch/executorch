# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from unittest.mock import patch

import pytest

from executorch.backends.apple.coreai.runtime.test import (
    run_sdk_source_lifetime as driver,
)


@pytest.mark.parametrize("architecture", ["unknown", "Unknown", "", "  "])
def test_compiled_variant_skipped_without_supported_architecture(
    tmp_path, capsys, architecture
):
    """AOT compile rejects unknown architectures, so only portable runs."""
    calls = []
    with (
        patch.object(driver, "run", return_value=architecture),
        patch.object(
            driver,
            "_export_weighted_fixture",
            side_effect=lambda d: d.mkdir(parents=True),
        ),
        patch.object(
            driver, "compile_source", side_effect=lambda *args: calls.append("compile")
        ),
        patch.object(
            driver,
            "run_variant",
            side_effect=lambda *args: calls.append(args[1]),
        ),
        patch.object(driver.tempfile, "TemporaryDirectory") as temporary,
    ):
        temporary.return_value.__enter__.return_value = str(tmp_path)
        driver.main(Path("probe"))

    assert calls == ["portable"]
    assert "compiled: skipped" in capsys.readouterr().out


def test_compiled_variant_runs_with_known_architecture(tmp_path):
    """Physical known-arch Macs still exercise the AOT path."""
    calls = []
    with (
        patch.object(driver, "run", return_value="h16c"),
        patch.object(
            driver,
            "_export_weighted_fixture",
            side_effect=lambda d: d.mkdir(parents=True),
        ),
        patch.object(
            driver, "compile_source", side_effect=lambda *args: calls.append("compile")
        ),
        patch.object(
            driver,
            "run_variant",
            side_effect=lambda *args: calls.append(args[1]),
        ),
        patch.object(driver.tempfile, "TemporaryDirectory") as temporary,
    ):
        temporary.return_value.__enter__.return_value = str(tmp_path)
        driver.main(Path("probe"))

    assert calls == ["portable", "compile", "compiled"]


@pytest.mark.parametrize(
    "failure", [RuntimeError("probe failed"), KeyboardInterrupt("interrupted")]
)
def test_cleanup_failure_preserves_original_error(tmp_path, capsys, failure):
    """A failing evict must not mask the probe failure it cleans up after."""

    def fake_run(command, expected_status=0):
        if "specialize" in command:
            raise failure
        raise RuntimeError("evict failed")

    bookmark = tmp_path / "saved.bookmark"
    bookmark.write_bytes(b"bookmark")
    with patch.object(driver, "run", side_effect=fake_run):
        with pytest.raises(type(failure)) as raised:
            driver.run_variant(
                Path("probe"),
                "portable",
                tmp_path / "model.aimodel",
                tmp_path / "fixture.json",
                bookmark,
            )

    assert raised.value is failure
    assert "cleanup evict failed: evict failed" in capsys.readouterr().out
