# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest

from executorch.backends.qualcomm.export_utils import (
    generate_gpu_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    QcomChipset,
)


def with_gpu_context(func):
    def wrapper(request, kwargs):
        preserved = {k: kwargs.pop(k) for k in ["expected"]}
        qnn_config = request.getfixturevalue("qnn_config")
        fixtures = {
            "quantizer": None,
            "compile_spec": generate_qnn_executorch_compiler_spec(
                soc_model=getattr(QcomChipset, qnn_config.soc_model),
                backend_options=generate_gpu_compiler_spec(),
                online_prepare=True,
            ),
        }
        return func(request, fixtures | preserved)

    return wrapper


def enumerate_fp_dtype(metric: Any):
    def wrapper(test_body):
        return pytest.mark.parametrize(
            "kwargs",
            [pytest.param({"act": None, "expected": metric}, id="fp")],
        )(test_body)

    return wrapper
