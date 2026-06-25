# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import pytest

from executorch.backends.arm._passes import RewriteConvPass
from executorch.backends.arm._passes.arm_pass_manager import (
    _registered_pass_insertions,
    clear_registered_pass_insertions,
    PassInsertions,
)
from executorch.backends.arm.vgf import backend as vgf_backend, VgfCompileSpec
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
