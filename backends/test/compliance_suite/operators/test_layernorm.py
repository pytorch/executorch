# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Model(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int]] = 10,
        eps=1e-5,
        elementwise_affine=True,
    ):
        super().__init__()
        self.ln = torch.nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        
    def forward(self, x):
        return self.ln(x)

@operator_test
class TestLayerNorm(OperatorTest):
    @dtype_test
    def test_layernorm_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, normalized_shape)
        self._test_op(Model().to(dtype), ((torch.rand(3, 10) * 10).to(dtype),), tester_factory)
        
    def test_layernorm_2d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, normalized_shape)
        self._test_op(Model(), (torch.randn(5, 10),), tester_factory)
        
    def test_layernorm_3d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, seq_len, normalized_shape)
        self._test_op(Model(), (torch.randn(5, 15, 10),), tester_factory)
        
    def test_layernorm_4d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, height, width) with normalization over last dimension
        self._test_op(Model(normalized_shape=8), (torch.randn(5, 10, 8, 8),), tester_factory)
        
    def test_layernorm_multidim_normalized_shape(self, tester_factory: Callable) -> None:
        # Normalize over last 2 dimensions
        self._test_op(Model(normalized_shape=[8, 8]), (torch.randn(5, 10, 8, 8),), tester_factory)
        
    def test_layernorm_multidim_normalized_shape_3d(self, tester_factory: Callable) -> None:
        # Normalize over last 3 dimensions
        self._test_op(Model(normalized_shape=[4, 4, 4]), (torch.randn(5, 4, 4, 4),), tester_factory)
        
    def test_layernorm_custom_eps(self, tester_factory: Callable) -> None:
        self._test_op(Model(eps=1e3), (torch.randn(5, 10),), tester_factory)
        
    def test_layernorm_no_elementwise_affine(self, tester_factory: Callable) -> None:
        self._test_op(Model(elementwise_affine=False), (torch.randn(5, 10),), tester_factory)
