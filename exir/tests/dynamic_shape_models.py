# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
from torch import nn

# TODO hack tracking here: T143942601
batch_norm_op = torch.ops.aten.native_batch_norm.default
if torch._C.DispatchKey.Autograd in batch_norm_op.py_kernels:
    del batch_norm_op.py_kernels[torch._C.DispatchKey.Autograd]
batch_norm_op._dispatch_cache.clear()


class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)

    def get_upper_bound_inputs(self):
        return (torch.rand(5, 3, 5, 5),)

    def get_random_inputs(self):
        bs = random.randint(2, 5)
        return (torch.rand(bs, 3, 5, 5),)

    @staticmethod
    def verify_graph(testcase: unittest.TestCase, graph_module: torch.fx.GraphModule):
        bn_node = [
            nd
            for nd in graph_module.graph.nodes
            if nd.target == torch.ops.aten.native_batch_norm.out
        ]
        testcase.assertEqual(1, len(bn_node))
        bn_node = bn_node[0]

        speclist = bn_node.meta["spec"]
        testcase.assertEqual(3, len(speclist))

        # for infernece, the save_mean and save_var should be empty
        _, save_mean_spec, save_var_spec = speclist
        testcase.assertEqual(list(save_mean_spec.shape), [0])
        testcase.assertEqual(list(save_var_spec.shape), [0])
