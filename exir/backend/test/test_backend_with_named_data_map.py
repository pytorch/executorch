# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend

from executorch.exir.backend.test.backend_with_named_data_map import (
    BackendWithNamedDataMap,
    BackendWithNDMPartitioner,
)


class TestBackendWithNamedDataMap(unittest.TestCase):
    def test_lowered_backend_module_has_output(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + x

        ep = to_edge(torch.export.export(M(), (torch.randn(1, 2),)))
        lowered = to_backend(
            BackendWithNamedDataMap.__name__, ep.exported_program(), []
        )

        buffer_entries = lowered.named_data_store_output.buffers
        self.assertTrue(len(buffer_entries) == 1)
        stored_data = lowered.named_data_store_output.pte_data

        self.assertTrue("aten.add.Tensor" in stored_data)
        self.assertTrue(buffer_entries[0] == bytes(1))

    def test_named_data_with_partitioner(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x + x
                y = torch.cos(y)
                y = y + y
                y = torch.sin(y)
                return y - y

        ep = to_edge(torch.export.export(M(), (torch.randn(1, 2),)))
        ep = ep.to_backend(BackendWithNDMPartitioner())

        ndm_output = ep._named_data_store.get_named_data_store_output()
        buffer_entries = ndm_output.buffers
        stored_data = ndm_output.pte_data
        self.assertEqual(len(buffer_entries), 3)
        self.assertTrue("aten.add.Tensor" in stored_data)
        self.assertTrue("aten.sub.Tensor" in stored_data)
        self.assertTrue("aten.sin.default" in stored_data)

    def test_named_data_with_control_flow(self):
        class M(torch.nn.Module):
            def true_branch(self, x):
                y = x * x
                y = torch.cos(y)
                return torch.sin(y)

            def false_branch(self, x):
                return torch.sin(x)

            def forward(self, x, y):
                z = x / y
                z = torch.cond(z.sum() > 0, self.true_branch, self.false_branch, [x])
                return z - z

        ep = to_edge(torch.export.export(M(), (torch.randn(1, 2), torch.randn(1, 2))))
        ep = ep.to_backend(BackendWithNDMPartitioner())

        ndm_output = ep._named_data_store.get_named_data_store_output()
        buffer_entries = ndm_output.buffers
        stored_data = ndm_output.pte_data
        self.assertEqual(len(buffer_entries), 4)
        self.assertTrue("aten.sub.Tensor" in stored_data)
        self.assertTrue("aten.div.Tensor" in stored_data)
        self.assertTrue("aten.sin.default" in stored_data)
        self.assertTrue("aten.mul.Tensor" in stored_data)
