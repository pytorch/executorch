# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir._serialize._serialize import serialize_for_executorch
from executorch.exir.scalar_type import ScalarType
from executorch.exir.tensor_layout import TensorLayout

from executorch.extension.flat_tensor.serialize.serialize import FlatTensorSerializer

class TestSerialize(unittest.TestCase):
    # Test serialize_for_executorch
    # When we have data in PTD
    # When we have NamedData in PTE
    # When we have TensorLayouts.
    # Also test pybindings.

    def test_linear(self) -> None:
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        config = ExecutorchBackendConfig(external_constants=True)
        model = to_edge(
            torch.export.export(LinearModule(), (torch.ones(5, 5),), strict=True)
        ).to_executorch(config=config)
        pte, ptds = serialize_for_executorch(model._emitter_output, config, FlatTensorSerializer(), named_data_store=model._named_data)

        self.assertEqual(len(ptds), 1)
        # Check that
        
