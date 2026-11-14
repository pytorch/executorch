# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from types import ModuleType, SimpleNamespace

import pytest
import torch

_WRAPPER_MODULE = "executorch.backends.cortex_m.library.cmsis_nn"
_BUNDLED_MODULE = "executorch.backends.cortex_m.library._cmsis_nn.cmsis_nn"
_BUNDLED_PACKAGE = "executorch.backends.cortex_m.library._cmsis_nn"
_EXTERNAL_MODULE = "cmsis_nn"


class _ExternalCmsisNNModule(ModuleType):
    Backend: SimpleNamespace
    CortexM: SimpleNamespace
    DataType: SimpleNamespace

    def __init__(self) -> None:
        super().__init__(_EXTERNAL_MODULE)
        self.Backend = SimpleNamespace(MVE="mve")
        self.CortexM = SimpleNamespace(M55="m55")
        self.DataType = SimpleNamespace(A8W8="a8w8")

    def resolve_backend(self, cpu: object) -> tuple[str, object]:
        return ("external", cpu)


def _import_cmsis_nn_wrapper():
    try:
        return importlib.import_module(_WRAPPER_MODULE)
    except Exception as exc:
        pytest.fail(f"Failed to resolve cmsis_nn: {exc}")


@pytest.fixture
def modify_available_cmsis_nn_modules(monkeypatch):
    """Returns a cmsis_nn wrapper with specified backing cmsis_nn packages.

    missing_name:
    """
    module = _import_cmsis_nn_wrapper()
    original_import_module = importlib.import_module

    def _reload(*, available_modules=()):
        def fake_import_module(name, package=None):
            if name == _WRAPPER_MODULE:
                return original_import_module(name, package)
            if name == _EXTERNAL_MODULE:
                if name in available_modules:
                    return _ExternalCmsisNNModule()
                raise ModuleNotFoundError(name=name)
            if name == _BUNDLED_MODULE:
                if name in available_modules:
                    return original_import_module(name, package)
                missing_name = (
                    _BUNDLED_MODULE
                    if _BUNDLED_PACKAGE in available_modules
                    else _BUNDLED_PACKAGE
                )
                raise ModuleNotFoundError(name=missing_name)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        return importlib.reload(module)

    yield _reload

    monkeypatch.undo()
    importlib.reload(module)


def test_cmsis_nn_convolve_wrapper_buffer_size() -> None:
    cmsis_nn = _import_cmsis_nn_wrapper()

    buf_size = cmsis_nn.convolve_wrapper_buffer_size(
        cmsis_nn.Backend.MVE,
        cmsis_nn.DataType.A8W8,
        input_nhwc=[1, 8, 8, 16],
        filter_nhwc=[8, 3, 3, 16],
        output_nhwc=[1, 6, 6, 8],
        padding_hw=[0, 0],
        stride_hw=[1, 1],
        dilation_hw=[1, 1],
    )

    assert buf_size == 576


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_vector_sum_s8_weights_layout(layout) -> None:
    cmsis_nn = _import_cmsis_nn_wrapper()
    if layout == "sliced":
        weights = torch.arange(-24, 24, dtype=torch.int8).reshape(4, 12)[:, ::2]
    else:
        weights = torch.arange(-12, 12, dtype=torch.int8).reshape(4, 6)
        if layout == "transposed":
            weights = weights.t().contiguous().t()
    assert weights.is_contiguous() == (layout == "contiguous")
    bias = torch.arange(4, dtype=torch.int32)
    input_offset = -7
    expected = weights.to(torch.int32).sum(dim=1, dtype=torch.int32) * input_offset
    expected += bias

    effective_bias = cmsis_nn.vector_sum_s8(weights, input_offset, bias)

    torch.testing.assert_close(effective_bias, expected)


def test_fallback_with_missing_module(
    modify_available_cmsis_nn_modules,
) -> None:
    cmsis_nn = modify_available_cmsis_nn_modules(
        available_modules=(_BUNDLED_PACKAGE, _EXTERNAL_MODULE),
    )

    assert cmsis_nn.resolve_backend(cmsis_nn.CortexM.M55) == ("external", "m55")


def test_fallback_with_missing_package(
    modify_available_cmsis_nn_modules,
) -> None:
    cmsis_nn = modify_available_cmsis_nn_modules(
        available_modules=(_EXTERNAL_MODULE),
    )

    assert cmsis_nn.resolve_backend(cmsis_nn.CortexM.M55) == ("external", "m55")


def test_bundled_module_preferred_over_external(
    modify_available_cmsis_nn_modules,
) -> None:
    cmsis_nn = modify_available_cmsis_nn_modules(
        available_modules=(_BUNDLED_MODULE, _BUNDLED_PACKAGE, _EXTERNAL_MODULE),
    )

    assert cmsis_nn.resolve_backend(cmsis_nn.CortexM.M55) != ("external", "m55")


def test_raise_without_dependency(modify_available_cmsis_nn_modules) -> None:
    cmsis_nn = modify_available_cmsis_nn_modules(
        available_modules=(),
    )

    with pytest.raises(
        ModuleNotFoundError,
        match="Cortex-M backend dependencies are not installed",
    ):
        cmsis_nn.resolve_backend(cmsis_nn.CortexM.M55)


@pytest.mark.parametrize(
    "kernel,padding,stride,dilation,expected",
    [
        ((1, 1), (0, 0), (1, 1), (1, 1), 0),
        ((1, 3), (0, 1), (1, 2), (1, 1), 128),
        ((1, 3), (0, 0), (1, 1), (1, 1), 128),
        ((1, 3), (0, 2), (1, 1), (1, 2), 128),
    ],
)
def test_packed_scratch_size(kernel, padding, stride, dilation, expected):
    cmsis_nn = _import_cmsis_nn_wrapper()
    size = cmsis_nn.convolve_wrapper_buffer_size(
        cmsis_nn.Backend.MVE,
        cmsis_nn.DataType.A8W8,
        input_nhwc=[1, 1, 9, 8],
        filter_nhwc=[7, *kernel, 8],
        output_nhwc=[1, 1, 9, 7],
        padding_hw=padding,
        stride_hw=stride,
        dilation_hw=dilation,
        weights_packed=True,
    )
    assert size == expected
