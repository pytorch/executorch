# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Model(torch.nn.Module):
    def __init__(
        self,
        num_embeddings=10,
        embedding_dim=5,
        padding_idx: Optional[int] = None,
        norm_type: float = 2.0,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            norm_type=norm_type,
        )
        
    def forward(self, x):
        return self.embedding(x)

@operator_test
class TestEmbedding(OperatorTest):
    @dtype_test
    def test_embedding_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, seq_length)
        # Note: Input indices should be of type Long (int64)
        model = Model().to(dtype)
        self._test_op(model, (torch.randint(0, 10, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        
    def test_embedding_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randint(0, 10, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        
    def test_embedding_sizes(self, tester_factory: Callable) -> None:
        # Test with different dictionary sizes and embedding dimensions
        self._test_op(Model(num_embeddings=5, embedding_dim=3), 
                     (torch.randint(0, 5, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        self._test_op(Model(num_embeddings=100, embedding_dim=10), 
                     (torch.randint(0, 100, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        self._test_op(Model(num_embeddings=1000, embedding_dim=50), 
                     (torch.randint(0, 1000, (2, 4), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        
    def test_embedding_padding_idx(self, tester_factory: Callable) -> None:
        # Test with padding_idx
        self._test_op(Model(padding_idx=0), 
                     (torch.randint(0, 10, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        self._test_op(Model(padding_idx=5), 
                     (torch.randint(0, 10, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)
        
    def test_embedding_input_shapes(self, tester_factory: Callable) -> None:
        # Test with different input shapes
        self._test_op(Model(), (torch.randint(0, 10, (5,), dtype=torch.long),), tester_factory, use_random_test_inputs=False)  # 1D input
        self._test_op(Model(), (torch.randint(0, 10, (2, 8), dtype=torch.long),), tester_factory, use_random_test_inputs=False)  # 2D input
        self._test_op(Model(), (torch.randint(0, 10, (2, 3, 4), dtype=torch.long),), tester_factory, use_random_test_inputs=False)  # 3D input
        