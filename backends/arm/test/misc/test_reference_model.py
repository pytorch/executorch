# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import tosa_reference_model
from executorch.backends.arm.test import runner_utils
from executorch.backends.arm.tosa import reference_model
from executorch.backends.arm.tosa.specification import TosaSpecification


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


def test_runner_utils_preserves_reference_model_imports() -> None:
    assert (
        runner_utils.TosaReferenceModelDispatch
        is reference_model.TosaReferenceModelDispatch
    )
    assert runner_utils.run_tosa_graph is reference_model.run_tosa_graph
    assert runner_utils.numpy_to_torch_tensor is reference_model.numpy_to_torch_tensor


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


def test_reference_model_status_must_be_valid(monkeypatch) -> None:
    monkeypatch.setattr(
        tosa_reference_model,
        "run",
        lambda *args, **kwargs: ([], object()),
    )
    spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    output = SimpleNamespace(args=((),))

    with pytest.raises(RuntimeError, match="rejected the serialized graph"):
        reference_model.run_tosa_graph(b"tosa", spec, [], cast(Any, output))


def test_reference_model_output_arity_must_match_graph(monkeypatch) -> None:
    monkeypatch.setattr(
        tosa_reference_model,
        "run",
        lambda *args, **kwargs: (
            [object(), object()],
            tosa_reference_model.GraphStatus.TOSA_VALID,
        ),
    )
    spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    output = SimpleNamespace(args=((object(),),))

    with pytest.raises(RuntimeError, match="output arity disagrees"):
        reference_model.run_tosa_graph(b"tosa", spec, [], cast(Any, output))


@pytest.mark.parametrize(
    ("storage_shape", "declared_shape", "dim_order", "inverse_order"),
    (
        ((1, 2, 3, 4), (1, 4, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)),
        ((1, 2, 3, 4), (1, object(), 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)),
        ((1, 2, 2, 3, 4), (1, 2, 4, 2, 3), (0, 1, 3, 4, 2), (0, 1, 4, 2, 3)),
        (
            (1, 2, 2, 3, 4),
            (1, 2, object(), 2, 3),
            (0, 1, 3, 4, 2),
            (0, 1, 4, 2, 3),
        ),
    ),
)
def test_bfloat16_outputs_keep_runtime_shape_before_layout_restore(
    monkeypatch,
    storage_shape,
    declared_shape,
    dim_order,
    inverse_order,
) -> None:
    output_tensor = SimpleNamespace(
        dtype=torch.bfloat16,
        shape=declared_shape,
        dim_order=lambda: dim_order,
    )
    monkeypatch.setattr(
        reference_model, "get_first_fake_tensor", lambda _node: output_tensor
    )
    words = np.arange(np.prod(storage_shape), dtype=np.uint16).reshape(storage_shape)

    result = reference_model.numpy_to_torch_tensor(
        words.view(np.dtype("V2")), cast(Any, object())
    )
    expected = torch.from_numpy(words).permute(inverse_order).contiguous()

    assert result.dtype == torch.bfloat16
    assert result.shape == expected.shape
    assert torch.equal(result.view(torch.uint16), expected)


def test_shape_inference_rejects_an_empty_executable() -> None:
    with pytest.raises(ValueError, match="must be nonempty"):
        reference_model.TosaReferenceModelDispatch("")


def test_shape_inference_uses_configured_executable(monkeypatch, tmp_path: Path) -> None:
    command: list[str] = []
    options: dict[str, object] = {}

    def run(args, **kwargs) -> None:
        command.extend(args)
        options.update(kwargs)
        (tmp_path / "resolved_model.tosa").write_bytes(b"resolved")

    monkeypatch.setattr(reference_model.subprocess, "run", run)
    dispatch = reference_model.TosaReferenceModelDispatch("custom-infer-shapes")

    result = dispatch._run_infer_shapes(b"model", [], (), tmp_path)

    assert result == b"resolved"
    assert command == ["custom-infer-shapes", str(tmp_path / "test_case.json")]
    assert options == {"check": True}


def test_shape_inference_requires_aligned_names_and_tensors(tmp_path: Path) -> None:
    dispatch = reference_model.TosaReferenceModelDispatch()

    with pytest.raises(ValueError, match="names and tensors must align"):
        dispatch._generate_shape_inference_json(
            tmp_path / "model.tosa",
            tmp_path / "test_case.json",
            ["input"],
            (),
        )


def test_shape_inference_method_override_is_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    command: list[str] = []

    def run(args, **_kwargs) -> None:
        command.extend(args)
        (tmp_path / "resolved_model.tosa").write_bytes(b"resolved")

    monkeypatch.setattr(reference_model.subprocess, "run", run)
    dispatch = reference_model.TosaReferenceModelDispatch("configured")

    assert (
        dispatch._run_infer_shapes(
            b"model", [], (), tmp_path, infer_shapes_path="override"
        )
        == b"resolved"
    )
    assert command[0] == "override"
    with pytest.raises(ValueError, match="must be nonempty"):
        dispatch._run_infer_shapes(b"model", [], (), tmp_path, infer_shapes_path="")


def test_shape_inference_requires_resolved_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reference_model.subprocess, "run", lambda *_args, **_kwargs: None)
    dispatch = reference_model.TosaReferenceModelDispatch()

    with pytest.raises(RuntimeError, match="without producing resolved_model.tosa"):
        dispatch._run_infer_shapes(b"model", [], (), tmp_path)
