# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from executorch.backends.xnnpack import get_xnnpack_recipe
from executorch.export import export
from torch.testing._internal.common_quantization import TestHelperModules

class TestXnnpackRecipes(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_basic_recipe(self) -> None:
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        export_session = export(
            model=m_eager, 
            example_inputs=example_inputs, 
            export_recipe=get_xnnpack_recipe("FP32_CPU_ACCELERATED_RECIPE")
        )
        export_session.export()
    
    def test_dynamic_quant_recipe(self) -> None:
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        export_session = export(
            model=m_eager, 
            example_inputs=example_inputs, 
            export_recipe=get_xnnpack_recipe("DYNAMIC_QUANT_CPU_ACCELERATED_RECIPE")
        )
        export_session.export()
