# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

import torch
from executorch.devtools.backend_debug import DelegationBreakdown, get_delegation_info
from executorch.exir import to_edge
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from pandas.testing import assert_frame_equal


class TestUtils(unittest.TestCase):
    def test_get_delegation_info(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = to_edge(torch.export.export(m, inputs)).to_backend(
            AddMulPartitionerDemo()
        )
        delegation_info = get_delegation_info(edge.exported_program().graph_module)

        self.assertEqual(delegation_info.num_delegated_subgraphs, 2)
        self.assertEqual(delegation_info.num_delegated_nodes, 4)
        self.assertEqual(delegation_info.num_non_delegated_nodes, 3)
        expected_delegation_by_op_dict = {
            "aten_add_tensor": DelegationBreakdown(
                op_type="aten_add_tensor", delegated=2, non_delegated=0
            ),
            "aten_mm_default": DelegationBreakdown(
                op_type="aten_mm_default", delegated=2, non_delegated=0
            ),
            "aten_sub_tensor": DelegationBreakdown(
                op_type="aten_sub_tensor", delegated=0, non_delegated=1
            ),
            "getitem": DelegationBreakdown(
                op_type="getitem", delegated=0, non_delegated=2
            ),
        }
        self.assertEqual(
            delegation_info.delegation_by_operator, expected_delegation_by_op_dict
        )

        self.assertIn(
            "Total delegated subgraphs",
            delegation_info.get_summary(),
        )

        df = delegation_info.get_operator_delegation_dataframe()
        expected_df = pd.DataFrame(
            {
                "op_type": [
                    "aten_add_tensor",
                    "aten_mm_default",
                    "aten_sub_tensor",
                    "getitem",
                    "Total",
                ],
                "occurrences_in_delegated_graphs": [2, 2, 0, 0, 4],
                "occurrences_in_non_delegated_graphs": [0, 0, 1, 2, 3],
            }
        )
        assert_frame_equal(expected_df, df)
