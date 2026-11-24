# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import pytest
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget


def _scratch_queries(target, kernel_mode):
    queries = get_scratch_queries(target, kernel_mode)
    if queries is None:
        if os.environ.get("EXECUTORCH_REQUIRE_ESP_NN_SCRATCH") == "1":
            pytest.fail(
                f"Required ESP-NN scratch binding missing: {target}/{kernel_mode}"
            )
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    return queries


@pytest.mark.parametrize("target", list(EspTarget))
@pytest.mark.parametrize("kernel_mode", list(EspKernelMode))
def test_vendor_scratch_queries(target, kernel_mode):
    queries = _scratch_queries(target, kernel_mode)
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
@pytest.mark.parametrize("value", [-1, 0, 65536])
def test_fc_dimension_bounds(dimension, value):
    queries = _scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
    args = {"batches": 8, "row_len": 16, "out_ch": 32}
    args[dimension] = value
    with pytest.raises(ValueError, match=dimension):
        queries.fc_batch_scratch_size(**args)


@pytest.fixture(
    params=[
        (EspTarget.ESP32S3, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32P4, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32S3, EspKernelMode.ANSI),
    ],
    ids=["s3", "pie", "generic", "ansi"],
)
def scratch_queries(request):
    return _scratch_queries(*request.param)


@pytest.mark.parametrize("depthwise", [False, True], ids=["conv", "depthwise"])
@pytest.mark.parametrize(
    "argument,value",
    [
        (argument, value)
        for argument in ("input_dims", "filter_dims", "output_dims", "stride")
        for value in (-1, 0, 65536)
    ]
    + [("padding", -1), ("padding", 65536)],
)
def test_convolution_dimension_bounds(scratch_queries, depthwise, argument, value):
    args = {
        "input_dims": (8, 8, 3),
        "filter_dims": (1, 1, 3),
        "output_dims": (8, 8, 16),
        "stride": (1, 1),
        "padding": (0, 0),
    }
    query = scratch_queries.conv2d_scratch_size
    if depthwise:
        query = scratch_queries.depthwise_conv2d_scratch_size
        args["output_dims"] = (8, 8, 3)
        args["channel_multiplier"] = 1
    for axis in range(len(args[argument])):
        invalid = list(args[argument])
        invalid[axis] = value
        with pytest.raises(ValueError, match=argument):
            query(**{**args, argument: tuple(invalid)})


def test_convolution_dimension_boundaries(scratch_queries):
    expected = {
        "_esp_nn_scratch_esp32s3": (192, 32),
        "_esp_nn_scratch_riscv_pie": (84, 0),
        "_esp_nn_scratch_generic": (0, 0),
        "_esp_nn_scratch_ansi": (0, 0),
    }[scratch_queries.__name__.rsplit(".", 1)[-1]]
    assert (
        scratch_queries.conv2d_scratch_size(
            input_dims=(65535, 1, 1),
            filter_dims=(1, 1, 1),
            output_dims=(65535, 1, 1),
            stride=(1, 1),
            padding=(0, 0),
        ),
        scratch_queries.depthwise_conv2d_scratch_size(
            input_dims=(1, 1, 1),
            filter_dims=(1, 1, 1),
            output_dims=(1, 1, 65535),
            stride=(1, 1),
            padding=(0, 0),
            channel_multiplier=65535,
        ),
    ) == expected


@pytest.mark.parametrize("depthwise", [False, True], ids=["conv", "depthwise"])
@pytest.mark.parametrize(
    "input_dims,filter_dims,output_dims,stride,padding",
    [
        ((32766, 32766, 2), (2, 2, 2), (32767, 32767, 1), (1, 1), (1, 1)),
        ((65534, 65534, 1), (2, 2, 1), (65535, 65535, 1), (1, 1), (1, 1)),
        ((1, 1, 16), (65535, 65535, 16), (1, 1, 16), (1, 1), (0, 0)),
        ((1, 1, 1), (2, 2, 1), (65535, 65535, 1), (65535, 65535), (0, 0)),
    ],
    ids=["negative_size", "positive_wrapped_size", "filter_product", "coordinates"],
)
def test_convolution_arithmetic_bounds(
    scratch_queries, depthwise, input_dims, filter_dims, output_dims, stride, padding
):
    args = {
        "input_dims": input_dims,
        "filter_dims": filter_dims,
        "output_dims": output_dims,
        "stride": stride,
        "padding": padding,
    }
    query = scratch_queries.conv2d_scratch_size
    if depthwise:
        query = scratch_queries.depthwise_conv2d_scratch_size
        args["channel_multiplier"] = 1
    with pytest.raises(ValueError, match="int32 range"):
        query(**args)


@pytest.mark.parametrize("width", [2**29, 2**30])
def test_softmax_arithmetic_bounds(scratch_queries, width):
    with pytest.raises(ValueError, match="int32 range"):
        scratch_queries.softmax_scratch_size(width=width, height=1)


@pytest.mark.parametrize("value", [-1, 0])
def test_scalar_dimension_bounds(scratch_queries, value):
    with pytest.raises(ValueError, match="softmax dimensions"):
        scratch_queries.softmax_scratch_size(width=value, height=1)
    with pytest.raises(ValueError, match="softmax dimensions"):
        scratch_queries.softmax_scratch_size(width=1, height=value)
    for axis in range(3):
        dims = [1, 1, 1]
        dims[axis] = value
        with pytest.raises(ValueError, match="mean dimensions"):
            scratch_queries.mean_scratch_size(*dims)
    with pytest.raises(ValueError, match="batches"):
        scratch_queries.fc_batch_scratch_size(batches=value, row_len=16, out_ch=32)


def test_fc_arithmetic_bounds(scratch_queries):
    with pytest.raises(ValueError, match="int32 range"):
        scratch_queries.fc_batch_scratch_size(batches=1, row_len=65535, out_ch=32768)


def test_fc_batches_without_scratch(scratch_queries):
    assert scratch_queries.fc_batch_scratch_size(65536, 16, 32) == 0


@pytest.mark.parametrize("value", [-1, 0, 65536])
def test_depthwise_multiplier_bounds(scratch_queries, value):
    with pytest.raises(ValueError, match="channel_multiplier"):
        scratch_queries.depthwise_conv2d_scratch_size(
            input_dims=(8, 8, 3),
            filter_dims=(1, 1, 3),
            output_dims=(8, 8, 3),
            stride=(1, 1),
            padding=(0, 0),
            channel_multiplier=value,
        )


def test_missing_binding(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
    )
    assert get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None


@pytest.mark.parametrize(
    "error",
    [
        ImportError("undefined symbol: esp_nn_get_conv_scratch_size_esp32s3"),
        ModuleNotFoundError("missing dependency", name="binding_dependency"),
    ],
)
def test_broken_binding_propagates_error(monkeypatch, error):
    def broken_import(name):
        raise error

    monkeypatch.setattr(
        "executorch.backends.espressif.scratch.importlib.import_module", broken_import
    )
    with pytest.raises(type(error), match=str(error)):
        get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)


def test_required_binding_cannot_skip(monkeypatch):
    monkeypatch.setenv("EXECUTORCH_REQUIRE_ESP_NN_SCRATCH", "1")
    monkeypatch.setitem(
        sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
    )
    with pytest.raises(pytest.fail.Exception, match="Required ESP-NN scratch binding"):
        _scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)


def test_invalid_kernel_mode():
    with pytest.raises(ValueError, match="EspKernelMode"):
        get_scratch_queries(EspTarget.ESP32S3, "unknown")


def test_invalid_target():
    with pytest.raises(ValueError, match="EspTarget"):
        get_scratch_queries("unknown", EspKernelMode.OPTIMIZED)
