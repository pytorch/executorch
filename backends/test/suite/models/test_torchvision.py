# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torchvision
import unittest

from executorch.backends.test.suite.models import model_test_params, model_test_cls, run_model_test
from torch.export import Dim
from typing import Callable

#
# This file contains model integration tests for supported torchvision models.
# 

@model_test_cls
class TorchVision(unittest.TestCase):
    def _test_cv_model(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        use_dynamic_shapes: bool,
        tester_factory: Callable,
    ):
        # Test a CV model that follows the standard conventions.
        inputs = (
            torch.randn(1, 3, 224, 224, dtype=dtype),
        )
            
        dynamic_shapes = (
            {
                2: Dim("height", min=1, max=16)*16,
                3: Dim("width", min=1, max=16)*16,
            },
        ) if use_dynamic_shapes else None
        
        run_model_test(model, inputs, dtype, dynamic_shapes, tester_factory)


    def test_alexnet(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.alexnet()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_convnext_small(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.convnext_small()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_densenet161(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.densenet161()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_efficientnet_b4(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.efficientnet_b4()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
        

    def test_efficientnet_v2_s(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.efficientnet_v2_s()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_googlenet(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.googlenet()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_inception_v3(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.inception_v3()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    @model_test_params(supports_dynamic_shapes=False)
    def test_maxvit_t(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.maxvit_t()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_mnasnet1_0(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.mnasnet1_0()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_mobilenet_v2(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.mobilenet_v2()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_mobilenet_v3_small(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.mobilenet_v3_small()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    def test_regnet_y_1_6gf(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.regnet_y_1_6gf()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    

    def test_resnet50(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.resnet50()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    
        
    def test_resnext50_32x4d(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.resnext50_32x4d()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    
        
    def test_shufflenet_v2_x1_0(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.shufflenet_v2_x1_0()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    
        
    def test_squeezenet1_1(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.squeezenet1_1()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    
        
    def test_swin_v2_t(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.swin_v2_t()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    
        
    def test_vgg11(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.vgg11()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)


    @model_test_params(supports_dynamic_shapes=False)
    def test_vit_b_16(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.vit_b_16()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
        

    def test_wide_resnet50_2(self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable):
        model = torchvision.models.wide_resnet50_2()
        self._test_cv_model(model, dtype, use_dynamic_shapes, tester_factory)
    