# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from executorch.backends.arm.tosa import reference_model


def test_dispatch_runs_only_tosa_backend_delegate(monkeypatch) -> None:
    dispatch = reference_model.TosaReferenceModelDispatch()
    lowered = SimpleNamespace(backend_id="TOSABackend")
    monkeypatch.setattr(
        dispatch, "_tosa_dispatch", lambda module, inputs: (module, inputs)
    )

    result = dispatch.__torch_function__(
        torch._higher_order_ops.executorch_call_delegate,
        (),
        (lowered, "input"),
    )

    assert result == (lowered, ("input",))
    assert dispatch.ran_tosa_dispatch


def test_dispatch_rejects_wrong_backend() -> None:
    dispatch = reference_model.TosaReferenceModelDispatch()
    lowered = SimpleNamespace(backend_id="OtherBackend")

    with pytest.raises(RuntimeError, match="backend_id='OtherBackend'"):
        dispatch.__torch_function__(
            torch._higher_order_ops.executorch_call_delegate,
            (),
            (lowered,),
        )


def test_dispatch_requires_a_tosa_delegate() -> None:
    with pytest.raises(
        RuntimeError,
        match="never ran TOSABackend delegate",
    ):
        with reference_model.TosaReferenceModelDispatch():
            pass


def test_run_tosa_graph_rejects_invalid_status(monkeypatch) -> None:
    fake_reference_model = MagicMock()
    fake_reference_model.GraphStatus.TOSA_VALID = object()
    fake_reference_model.run.return_value = ([], object())
    monkeypatch.setitem(sys.modules, "tosa_reference_model", fake_reference_model)

    with pytest.raises(RuntimeError, match="Non-valid TOSA"):
        reference_model.run_tosa_graph(
            graph=object(),
            tosa_version=reference_model.TosaSpecification.create_from_string(
                "TOSA-1.0+INT"
            ),
            inputs=(),
            output_node=MagicMock(),
        )
