# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import pytest

import torch
import torchvision
from executorch.backends.test.suite import dtype_to_str

from torch.export import Dim

#
# This file contains model integration tests for supported torchvision models. This
# suite intends to include all export-compatible torchvision models. For models with
# multiple size variants, one small or medium variant is used.
#

PARAMETERIZE_DTYPE = pytest.mark.parametrize("dtype", [torch.float32], ids=dtype_to_str)
PARAMETERIZE_DYNAMIC_SHAPES = pytest.mark.parametrize(
    "use_dynamic_shapes", [False, True], ids=["static_shapes", "dynamic_shapes"]
)
PARAMETERIZE_STATIC_ONLY = pytest.mark.parametrize(
    "use_dynamic_shapes", [False], ids=["static_shapes"]
)


def _test_cv_model(
    model: torch.nn.Module,
    test_runner,
    dtype: torch.dtype,
    use_dynamic_shapes: bool,
):
    model = model.eval().to(dtype)

    # Test a CV model that follows the standard conventions.
    inputs = (torch.randn(1, 3, 224, 224, dtype=dtype),)

    dynamic_shapes = (
        (
            {
                2: Dim("height", min=1, max=16) * 16,
                3: Dim("width", min=1, max=16) * 16,
            },
        )
        if use_dynamic_shapes
        else None
    )

    test_runner.lower_and_run_model(model, inputs, dynamic_shapes=dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_alexnet(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.alexnet()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_convnext_small(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.convnext_small()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_densenet161(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.densenet161()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_efficientnet_b4(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.efficientnet_b4()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_efficientnet_v2_s(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.efficientnet_v2_s()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_googlenet(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.googlenet()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_inception_v3(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.inception_v3()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_STATIC_ONLY
def test_maxvit_t(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.maxvit_t()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_mnasnet1_0(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.mnasnet1_0()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_mobilenet_v2(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.mobilenet_v2()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_mobilenet_v3_small(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.mobilenet_v3_small()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_regnet_y_1_6gf(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.regnet_y_1_6gf()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_resnet50(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.resnet50()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_resnext50_32x4d(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.resnext50_32x4d()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_shufflenet_v2_x1_0(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.shufflenet_v2_x1_0()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_squeezenet1_1(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.squeezenet1_1()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_swin_v2_t(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.swin_v2_t()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_vgg11(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.vgg11()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_STATIC_ONLY
def test_vit_b_16(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.vit_b_16()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)


@PARAMETERIZE_DTYPE
@PARAMETERIZE_DYNAMIC_SHAPES
def test_wide_resnet50_2(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchvision.models.wide_resnet50_2()
    _test_cv_model(model, test_runner, dtype, use_dynamic_shapes)
