# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Any, List

import pytest

from executorch.backends.qualcomm.export_utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    make_quantizer,
    QcomChipset,
    QnnExecuTorchBackendType,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import HtpArch


def with_htp_context(func, hw_arch):
    def wrapper(request, kwargs):
        # extend this if necessary
        preserved = {k: kwargs.pop(k) for k in ["expected"]}
        callbacks_and_args = {
            # extract objects from callback
            "quantizers": {"arch": hw_arch} | kwargs,
            "compile_specs": {"arch": hw_arch},
        }
        fixtures = {
            # e.g. "quantizer": quantizers(v68, act, ...)
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
                for i, (act, id) in enumerate([(8, "8a"), (16, "16a"), (None, "fp")])
            ],
        )(test_body)

    return wrapper


def _get_htp_arch():
    # hardcoded htp architecture with corresponding premium soc
    return [
        (HtpArch.V68, "SM8350"),
        (HtpArch.V69, "SM8450"),
        (HtpArch.V73, "SM8550"),
        (HtpArch.V75, "SM8650"),
        (HtpArch.V79, "SM8750"),
        (HtpArch.V81, "SM8850"),
    ]


@pytest.fixture(scope="session")
def quantizers():
    arch_to_soc = dict(_get_htp_arch())

    @lru_cache()
    def _build(arch, act, param, per_ch, per_block):
        attr = f"use_{act}a{param}w" + ("_block" if per_block else "")
        if quant_dtype := getattr(QuantDtype, attr, None):
            return make_quantizer(
                quant_dtype=quant_dtype,
                per_channel_conv=per_ch,
                per_channel_linear=per_ch,
                per_channel_embedding=per_ch,
                backend=QnnExecuTorchBackendType.kHtpBackend,
                soc_model=arch_to_soc[arch],
            )

    def get_quantizer(arch, act, param=None, pcq=False, lpbq=False, block_sz_map=None):
        param = 8 if (param is None and act is not None) else param
        if quantizer := _build(arch, act, param, pcq, lpbq):
            quantizer.set_block_size_map(block_sz_map or {})
        return quantizer

    return get_quantizer


@pytest.fixture(scope="session")
def compile_specs():
    compile_spec = {
        arch: generate_qnn_executorch_compiler_spec(
            soc_model=getattr(QcomChipset, soc_model),
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        for (arch, soc_model) in _get_htp_arch()
    }
    return lambda arch: compile_spec[arch]
