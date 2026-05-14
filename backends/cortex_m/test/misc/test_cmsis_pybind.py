# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest


def _import_cmsis_nn():
    try:
        return importlib.import_module("cmsis_nn")
    except Exception as exc:
        pytest.fail(f"Failed to resolve cmsis_nn: {exc}")


def test_cmsis_nn_convolve_wrapper_buffer_size() -> None:
    cmsis_nn = _import_cmsis_nn()

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
