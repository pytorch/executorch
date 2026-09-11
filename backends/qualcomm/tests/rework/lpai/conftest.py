# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Any, List

import pytest

from executorch.backends.qualcomm.export_utils import (
    generate_lpai_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    make_quantizer,
    QcomChipset,
    QnnExecuTorchBackendType,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    LpaiHardwareVersion,
    QnnExecuTorchLpaiTargetEnv,
)


def with_lpai_context(func, hw_arch):
    def wrapper(request, kwargs):
        # extend this if necessary
        preserved = {k: kwargs.pop(k) for k in ["expected"]}
        callbacks_and_args = {
            # extract objects from callback
            "quantizers": {"arch": hw_arch} | kwargs,
            "compile_specs": {"arch": hw_arch},
        }
        fixtures = {
            k[:-1]: request.getfixturevalue(k)(**v)
            for k, v in callbacks_and_args.items()
        }
        return func(request, fixtures | preserved)

    return wrapper


def enumerate_activation_dtype(metrics: List[Any]):
    def wrapper(test_body):
        return pytest.mark.parametrize(
            "kwargs",
            [
                pytest.param({"act": act, "expected": metrics[i]}, id=id)
                for i, (act, id) in enumerate(
                    [
                        (8, "8a"),
                    ]
                )
            ],
        )(test_body)

    return wrapper


def _get_lpai_arch():
    # hardcoded lpai architecture with corresponding premium soc
    return [
        (LpaiHardwareVersion.V6, "SM8850"),
    ]


@pytest.fixture(scope="session")
def quantizers():
    arch_to_soc = dict(_get_lpai_arch())

    @lru_cache()
    def _build(arch, act, param, per_ch):
        attr = f"use_{act}a{param}w"
        if quant_dtype := getattr(QuantDtype, attr, None):
            return make_quantizer(
                quant_dtype=quant_dtype,
                per_channel_conv=per_ch,
                per_channel_linear=per_ch,
                backend=QnnExecuTorchBackendType.kLpaiBackend,
                soc_model=arch_to_soc[arch],
            )

    def get_quantizer(arch, act, param=None, pcq=False, **_):
        param = 8 if (param is None and act is not None) else param
        return _build(arch, act, param, pcq)

    return get_quantizer


@pytest.fixture(scope="session")
def compile_specs():
    compile_spec = {
        arch: generate_qnn_executorch_compiler_spec(
            soc_model=getattr(QcomChipset, soc_model),
            backend_options=generate_lpai_compiler_spec(
                target_env=QnnExecuTorchLpaiTargetEnv.kX86,
            ),
        )
        for (arch, soc_model) in _get_lpai_arch()
    }
    return lambda arch: compile_spec[arch]
