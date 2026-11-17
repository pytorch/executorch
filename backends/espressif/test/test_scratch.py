# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import pytest
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget


@pytest.mark.parametrize("target", list(EspTarget))
@pytest.mark.parametrize("kernel_mode", list(EspKernelMode))
def test_vendor_scratch_queries(target, kernel_mode):
    queries = get_scratch_queries(target, kernel_mode)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    if kernel_mode == EspKernelMode.ANSI:
        expected = (0, 0, 0, 0, 0, 0, 256)
    elif target == EspTarget.ESP32S3:
        expected = (4928, 32, 2064, 0, 512, 704, 256)
    elif target in (EspTarget.ESP32P4, EspTarget.ESP32S31):
        expected = (1808, 0, 2064, 0, 0, 800, 256)
    else:
        expected = (0, 0, 128, 0, 0, 0, 256)
    conv_args = {
        "input_dims": (11, 7, 16),
        "filter_dims": (3, 2, 16),
        "output_dims": (6, 4, 32),
        "stride": (2, 2),
        "padding": (1, 1),
    }
    assert (
        queries.conv2d_scratch_size(**conv_args),
        queries.depthwise_conv2d_scratch_size(**conv_args, channel_multiplier=2),
        queries.softmax_scratch_size(width=32, height=8),
        queries.mean_scratch_size(height=8, width=8, channels=16),
        queries.hard_swish_scratch_size(),
        queries.fc_batch_scratch_size(batches=8, row_len=16, out_ch=32),
        queries.logistic_scratch_size(),
    ) == expected


@pytest.mark.parametrize("dimension", ["row_len", "out_ch"])
@pytest.mark.parametrize("value", [-1, 65536])
def test_fc_dimension_bounds(dimension, value):
    queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    args = {"batches": 8, "row_len": 16, "out_ch": 32}
    args[dimension] = value
    with pytest.raises(TypeError):
        queries.fc_batch_scratch_size(**args)


def test_missing_binding(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
    )
    assert get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None


def test_invalid_kernel_mode():
    with pytest.raises(ValueError, match="EspKernelMode"):
        get_scratch_queries(EspTarget.ESP32S3, "unknown")


def test_invalid_target():
    with pytest.raises(ValueError, match="EspTarget"):
        get_scratch_queries("unknown", EspKernelMode.OPTIMIZED)
