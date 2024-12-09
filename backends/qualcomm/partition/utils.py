# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

from executorch.exir.backend.compile_spec_schema import CompileSpec


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> bytes:
    for compiler_spec in compiler_specs:
        if compiler_spec.key == QCOM_QNN_COMPILE_SPEC:
            qnn_compile_spec_buffer = compiler_spec.value
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")
    return qnn_compile_spec_buffer
