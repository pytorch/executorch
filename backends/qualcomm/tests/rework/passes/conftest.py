# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import lru_cache

import pytest

from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.rework.conftest import default_property
from executorch.backends.qualcomm.tests.rework.passes.passes_helper import (
    Assertions,
    PassPipeline,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_lpai_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)

# Skip DSP
_DEFAULT_BACKENDS = [
    QnnExecuTorchBackendType.kGpuBackend,
    QnnExecuTorchBackendType.kHtpBackend,
    QnnExecuTorchBackendType.kLpaiBackend,
]


def _build_compile_specs() -> dict:
    backend_compiler_info = [
        (
            QnnExecuTorchBackendType.kHtpBackend,
            generate_htp_compiler_spec,
            {"use_fp16": True},
        ),
        (QnnExecuTorchBackendType.kGpuBackend, generate_gpu_compiler_spec, {}),
        (QnnExecuTorchBackendType.kLpaiBackend, generate_lpai_compiler_spec, {}),
    ]
    backend_compiler_specs = {}
    for backend_type, func, extra_args in backend_compiler_info:
        try:
            backend_compiler_specs[backend_type] = (
                generate_qnn_executorch_compiler_spec(
                    soc_model=getattr(QcomChipset, default_property().soc_model),
                    backend_options=func(**extra_args),
                )
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"compile_spec for {backend_type} unavailable: {e!r}"
            )
    return backend_compiler_specs


_COMPILE_SPEC_MAP = _build_compile_specs()


@pytest.fixture
def assertions():
    return Assertions


@pytest.fixture
def pass_pipeline():
    return PassPipeline


@pytest.fixture(scope="session")
def compile_specs():
    # If QNN version does not support the backend, it will skip. E.g. LPAI with older QNN version.
    return lambda backend_type: _COMPILE_SPEC_MAP.get(backend_type)


@pytest.fixture(scope="session")
def quantizers():
    """
    Factory that builds a QnnQuantizer.
    Returns None for GPU (no quantization rules yet) and htp_fp.
    """

    @lru_cache()
    def _build(backend_type, fp16):
        if fp16 or backend_type == QnnExecuTorchBackendType.kGpuBackend:
            return None
        return make_quantizer(backend=backend_type)

    return _build


_ParsedEntry = tuple[QnnExecuTorchBackendType, bool, str]


def _expand_entry(entry: QnnExecuTorchBackendType) -> list[_ParsedEntry]:
    """Expands kHtpBackend into both quantized and fp16 variants."""
    if not isinstance(entry, QnnExecuTorchBackendType):
        raise TypeError(f"expected QnnExecuTorchBackendType, got {type(entry)}")
    if entry == QnnExecuTorchBackendType.kUndefinedBackend:
        raise ValueError("kUndefinedBackend is not a valid backend")

    if entry == QnnExecuTorchBackendType.kHtpBackend:
        return [
            (entry, False, str(entry)),
            (entry, True, str(entry) + "_fp16"),
        ]
    return [(entry, False, str(entry))]


def enumerate_backends(backends: list[QnnExecuTorchBackendType] | None = None):
    """
    Parametrize a pass test over backends. Defaults to all supported backends.
    kHtpBackend auto-expands into [htp] (quantized) and [htp_fp16].

    Example:
      @enumerate_backends()                         # → [gpu], [htp], [htp_fp16], [lpai]
      @enumerate_backends([kHtpBackend])            # → [htp], [htp_fp16]
    """
    if backends is None:
        backends = _DEFAULT_BACKENDS
    if not isinstance(backends, list) or not backends:
        raise TypeError("enumerate_backends requires a non-empty list")
    entries = []
    for b in backends:
        entries.extend(_expand_entry(b))
    return pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"backend_type": bt, "fp16": fp16}, id=id_)
            for bt, fp16, id_ in entries
        ],
    )


def enumerate_backends_quantized():
    """
    Parametrize a pass test over backends whose fixture provides a quantizer.
    Excludes GPU (no quantization rules) and htp_fp16 (fp path, no quantizer).
    Use for passes that only fire on quantized graphs (e.g. InsertRequantize).
    """
    entries = [
        (
            QnnExecuTorchBackendType.kHtpBackend,
            False,
            str(QnnExecuTorchBackendType.kHtpBackend),
        ),
        (
            QnnExecuTorchBackendType.kLpaiBackend,
            False,
            str(QnnExecuTorchBackendType.kLpaiBackend),
        ),
    ]
    return pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"backend_type": bt, "fp16": fp16}, id=id_)
            for bt, fp16, id_ in entries
        ],
    )


def repack_pass_fixtures(func):
    def wrapper(request, kwargs):
        assert "backend_type" in kwargs, "Expecting backend_type to be in kwargs."
        backend_type = kwargs["backend_type"]
        fp16 = kwargs.pop("fp16", False)
        compile_spec = request.getfixturevalue("compile_specs")(backend_type)
        if compile_spec is None:
            pytest.skip(
                f"compile_spec for backend '{backend_type}' is not available "
                f"in the current QNN SDK"
            )
        kwargs["quantizer"] = request.getfixturevalue("quantizers")(backend_type, fp16)
        kwargs["compile_spec"] = compile_spec
        return func(request, kwargs)

    return wrapper
