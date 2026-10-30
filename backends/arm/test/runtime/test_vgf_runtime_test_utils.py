# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test.runtime import _vgf_runtime_test_utils as utils


def _make_executable(path: Path) -> None:
    path.write_bytes(b"")
    path.chmod(0o755)


def test_vgf_dump_is_resolved_next_to_model_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_dir = tmp_path / "trusted_tools"
    trusted_dir.mkdir()

    model_converter = trusted_dir / "model-converter"
    trusted_vgf_dump = trusted_dir / "vgf_dump"

    _make_executable(model_converter)
    _make_executable(trusted_vgf_dump)

    monkeypatch.setattr(
        utils,
        "require_model_converter_executable",
        lambda: model_converter,
    )

    assert utils._require_vgf_dump_executable() == trusted_vgf_dump.resolve()


def test_load_vgf_json_does_not_select_vgf_dump_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_dir = tmp_path / "trusted_tools"
    trusted_dir.mkdir()

    model_converter = trusted_dir / "model-converter"
    trusted_vgf_dump = trusted_dir / "vgf_dump"

    _make_executable(model_converter)
    _make_executable(trusted_vgf_dump)

    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()

    attacker_vgf_dump = attacker_dir / "vgf_dump"
    _make_executable(attacker_vgf_dump)

    # A malicious vgf_dump earlier in PATH must not affect tool selection.
    monkeypatch.setenv("PATH", str(attacker_dir))
    monkeypatch.setattr(
        utils,
        "require_model_converter_executable",
        lambda: model_converter,
    )

    vgf_path = tmp_path / "model.vgf"
    vgf_path.write_bytes(b"vgf")

    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> None:
        commands.append(command)

        assert check
        assert command[0] == str(trusted_vgf_dump.resolve())
        assert command[0] != str(attacker_vgf_dump.resolve())

        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text('{"trusted": true}')

        return None

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    result = utils.load_vgf_json(tmp_path)

    assert result == {"trusted": True}
    assert len(commands) == 1


def test_load_vgf_json_rejects_missing_sibling_vgf_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_dir = tmp_path / "trusted_tools"
    trusted_dir.mkdir()

    model_converter = trusted_dir / "model-converter"
    _make_executable(model_converter)

    monkeypatch.setattr(
        utils,
        "require_model_converter_executable",
        lambda: model_converter,
    )

    (tmp_path / "model.vgf").write_bytes(b"vgf")

    with pytest.raises(
        RuntimeError,
        match=r"Unable to locate an executable `vgf_dump`",
    ):
        utils.load_vgf_json(tmp_path)
