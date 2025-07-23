# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
import torchvision

from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.models import (
    model_test_cls,
    model_test_params,
    run_model_test,
)
from torch.export import Dim

#
# This file contains model integration tests for supported torchvision models. This
# suite intends to include all export-compatible torchvision models. For models with
# multiple size variants, one small or medium variant is used.
#


@model_test_cls
class TorchVision(unittest.TestCase):
    def _test_cv_model(
        self,
        model: torch.nn.Module,
        flow: TestFlow,
        dtype: torch.dtype,
        use_dynamic_shapes: bool,
    ):
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

        run_model_test(model, inputs, flow, dtype, dynamic_shapes)

    def test_alexnet(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.alexnet()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_convnext_small(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.convnext_small()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_densenet161(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.densenet161()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_efficientnet_b4(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.efficientnet_b4()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_efficientnet_v2_s(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.efficientnet_v2_s()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_googlenet(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.googlenet()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_inception_v3(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.inception_v3()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    @model_test_params(supports_dynamic_shapes=False)
    def test_maxvit_t(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.maxvit_t()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_mnasnet1_0(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.mnasnet1_0()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_mobilenet_v2(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.mobilenet_v2()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_mobilenet_v3_small(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.mobilenet_v3_small()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_regnet_y_1_6gf(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.regnet_y_1_6gf()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_resnet50(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.resnet50()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_resnext50_32x4d(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.resnext50_32x4d()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_shufflenet_v2_x1_0(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.shufflenet_v2_x1_0()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_squeezenet1_1(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.squeezenet1_1()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_swin_v2_t(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.swin_v2_t()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_vgg11(self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool):
        model = torchvision.models.vgg11()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    @model_test_params(supports_dynamic_shapes=False)
    def test_vit_b_16(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.vit_b_16()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)

    def test_wide_resnet50_2(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = torchvision.models.wide_resnet50_2()
        self._test_cv_model(model, flow, dtype, use_dynamic_shapes)
