# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from executorch.backends.arm._passes import RewriteConvPass
from executorch.backends.arm._passes.arm_pass_manager import (
    _registered_pass_insertions,
    clear_registered_pass_insertions,
    PassInsertions,
)

from executorch.backends.arm.vgf import backend, backend as vgf_backend, VgfCompileSpec
from executorch.backends.arm.vgf.backend import (
    _copy_failure_artifacts,
    _format_repro_command,
    _replace_converter_input_path,
    vgf_compile,
)
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.pass_base import ExportPass
from torch.export.exported_program import ExportedProgram
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class DummyPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


def _registry_state() -> dict[type, tuple[list[type], list[type]]]:
    return {
        pass_type: (
            [type(pass_) for pass_ in insertions.before_passes],
            [type(pass_) for pass_ in insertions.after_passes],
        )
        for pass_type, insertions in _registered_pass_insertions.items()
    }


def _set_up_fake_vgf_preprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        vgf_backend.TOSABackend,
        "filter_tosa_compile_specs",
        lambda compile_spec: [],
    )
    monkeypatch.setattr(
        vgf_backend,
        "arm_get_first_delegation_tag",
        lambda graph_module: "",
    )
    monkeypatch.setattr(
        vgf_backend.VgfBackend,
        "_compile_tosa_flatbuffer",
        staticmethod(lambda tosa_flatbuffer, compile_spec, tag_name="": b"vgf"),
    )


def _fake_exported_program() -> ExportedProgram:
    return cast(ExportedProgram, SimpleNamespace(graph_module=None))


def test_vgf_preprocess_restores_pass_registry(monkeypatch) -> None:
    clear_registered_pass_insertions()
    try:
        _registered_pass_insertions[RewriteConvPass] = PassInsertions(
            before_passes=[DummyPass()],
        )
        original_registry = _registry_state()
        _set_up_fake_vgf_preprocess(monkeypatch)
        monkeypatch.setattr(
            vgf_backend.TOSABackend,
            "_preprocess",
            lambda edge_program, compile_specs: PreprocessResult(processed_bytes=b""),
        )

        result = vgf_backend.VgfBackend.preprocess(
            _fake_exported_program(), VgfCompileSpec()._to_list()
        )

        assert result.processed_bytes == b"vgf"
        assert _registry_state() == original_registry
    finally:
        clear_registered_pass_insertions()


def test_vgf_preprocess_restores_pass_registry_on_failure(monkeypatch) -> None:
    clear_registered_pass_insertions()
    try:
        _registered_pass_insertions[RewriteConvPass] = PassInsertions(
            before_passes=[DummyPass()],
        )
        original_registry = _registry_state()
        _set_up_fake_vgf_preprocess(monkeypatch)

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(vgf_backend.TOSABackend, "_preprocess", _raise)

        with pytest.raises(RuntimeError, match="boom"):
            vgf_backend.VgfBackend.preprocess(
                _fake_exported_program(), VgfCompileSpec()._to_list()
            )

        assert _registry_state() == original_registry
    finally:
        clear_registered_pass_insertions()


def test_format_repro_command_quotes_shell_metacharacters():
    command = [
        "model-converter",
        "--flag=value with spaces",
        "-i",
        "input file.tosa",
        "-o",
        "output file.vgf",
    ]

    formatted = _format_repro_command(command)

    assert formatted == (
        "model-converter "
        "'--flag=value with spaces' "
        "-i "
        "'input file.tosa' "
        "-o "
        "'output file.vgf'"
    )


def test_replace_converter_input_path_replaces_input_after_i():
    command = [
        "model-converter",
        "--some-flag",
        "-i",
        "original.tosa",
        "-o",
        "output.vgf",
    ]

    replaced = _replace_converter_input_path(command, "preserved.tosa")

    assert replaced == [
        "model-converter",
        "--some-flag",
        "-i",
        "preserved.tosa",
        "-o",
        "output.vgf",
    ]
    assert command[3] == "original.tosa"


def test_copy_failure_artifacts_returns_none_without_artifact_path(tmp_path):
    tosa_path = tmp_path / "input.tosa"
    tosa_path.write_bytes(b"tosa bytes")

    copied_path = _copy_failure_artifacts(
        str(tosa_path),
        artifact_path=None,
        tag_name="delegate_0",
    )

    assert copied_path is None


def test_copy_failure_artifacts_copies_tosa_with_tag_name(tmp_path):
    tosa_path = tmp_path / "input.tosa"
    artifact_path = tmp_path / "artifacts"
    tosa_path.write_bytes(b"tosa bytes")

    copied_path = _copy_failure_artifacts(
        str(tosa_path),
        str(artifact_path),
        tag_name="delegate_0",
    )

    assert copied_path == os.path.join(
        str(artifact_path),
        "failed_model_converter_input_delegate_0.tosa",
    )
    assert os.path.exists(copied_path)
    assert open(copied_path, "rb").read() == b"tosa bytes"


def test_copy_failure_artifacts_copies_tosa_without_tag_name(tmp_path):
    tosa_path = tmp_path / "input.tosa"
    artifact_path = tmp_path / "artifacts"
    tosa_path.write_bytes(b"tosa bytes")

    copied_path = _copy_failure_artifacts(
        str(tosa_path),
        str(artifact_path),
        tag_name="",
    )

    assert copied_path == os.path.join(
        str(artifact_path),
        "failed_model_converter_input.tosa",
    )
    assert os.path.exists(copied_path)
    assert open(copied_path, "rb").read() == b"tosa bytes"


@mock.patch("executorch.backends.arm.vgf.backend.model_converter_env")
@mock.patch("executorch.backends.arm.vgf.backend.require_model_converter_binary")
@mock.patch("executorch.backends.arm.vgf.backend.subprocess.run")
def test_vgf_compile_failure_includes_repro_command_and_copies_tosa(
    mock_run,
    mock_require_model_converter_binary,
    mock_model_converter_env,
    tmp_path,
):
    artifact_path = tmp_path / "artifacts"

    mock_require_model_converter_binary.return_value = "model-converter"
    mock_model_converter_env.return_value = {"PATH": "/test/bin"}
    mock_run.side_effect = backend.subprocess.CalledProcessError(
        returncode=1,
        cmd=["model-converter"],
        output=b"converter stdout",
        stderr=b"converter stderr",
    )

    with pytest.raises(RuntimeError) as exc_info:
        vgf_compile(
            b"serialized tosa",
            ["--flag=value with spaces"],
            artifact_path=str(artifact_path),
            tag_name="delegate_0",
        )

    copied_tosa_path = os.path.join(
        str(artifact_path),
        "failed_model_converter_input_delegate_0.tosa",
    )

    assert os.path.exists(copied_tosa_path)
    assert open(copied_tosa_path, "rb").read() == b"serialized tosa"

    error = str(exc_info.value)
    assert "Vgf compiler failed." in error
    assert "Repro command:" in error
    assert "model-converter '--flag=value with spaces' -i" in error
    assert copied_tosa_path in error
    assert " -o " in error
    assert "Stderr:\nconverter stderr" in error
    assert "Stdout:\nconverter stdout" in error


@mock.patch("executorch.backends.arm.vgf.backend.model_converter_env")
@mock.patch("executorch.backends.arm.vgf.backend.require_model_converter_binary")
@mock.patch("executorch.backends.arm.vgf.backend.subprocess.run")
def test_vgf_compile_failure_includes_temp_repro_command_without_artifact_path(
    mock_run,
    mock_require_model_converter_binary,
    mock_model_converter_env,
):
    mock_require_model_converter_binary.return_value = "model-converter"
    mock_model_converter_env.return_value = {"PATH": "/test/bin"}
    mock_run.side_effect = backend.subprocess.CalledProcessError(
        returncode=1,
        cmd=["model-converter"],
        output=b"converter stdout",
        stderr=b"converter stderr",
    )

    with pytest.raises(RuntimeError) as exc_info:
        vgf_compile(
            b"serialized tosa",
            ["--some-flag"],
            artifact_path=None,
            tag_name="delegate_0",
        )

    error = str(exc_info.value)
    assert "Vgf compiler failed." in error
    assert "Repro command:" in error
    assert "model-converter --some-flag -i" in error
    assert "output_delegate_0.tosa.vgf" in error
    assert "failed_model_converter_input_delegate_0.tosa" not in error
    assert "Stderr:\nconverter stderr" in error
    assert "Stdout:\nconverter stdout" in error
