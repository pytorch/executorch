# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import Module  # @manual


class FTCondBasic(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        def true_branch(x):
            return x + x

        def false_branch(x):
            return x * x

        return torch.ops.higher_order.cond(
            inp.sum() > 4, true_branch, false_branch, [inp]
        )

    def get_random_inputs(self):
        return (torch.rand(5),)


class FTCondDynShape(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        def true_branch(x):
            return x + x + x

        def false_branch(x):
            return x * x * x

        return torch.ops.higher_order.cond(
            inp.sum() > 4, true_branch, false_branch, [inp]
        )

    def get_upper_bound_inputs(self):
        return (torch.rand(8),)

    def get_random_inputs(self):
        return (torch.rand(5),)


class FTCondDeadCode(Module):
    """
    A toy model used to test DCE on sub modules.

    The graph generated for torch.inverse will contain a node:
      torch.ops.aten._linalg_check_errors.default
    to check for errors. There are no out variants for this op and executorch
    runtime does not support it. For now, we simply erase this node by DCE
    since the Fx code does not consider this node as having side effect.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        def true_branch(x):
            x - 1
            return x + 1

        def false_branch(x):
            return x * 2

        return torch.ops.higher_order.cond(
            inp.sum() > 4, true_branch, false_branch, [inp]
        )

    def get_random_inputs(self):
        return (torch.eye(5) * 2,)


class FTMapBasic(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, y):
        def f(x, y):
            return x + y

        return torch.ops.higher_order.map(f, xs, y) + xs

    def get_random_inputs(self):
        return torch.rand(2, 4), torch.rand(4)


class FTMapDynShape(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, y):
        def f(x, y):
            return x + y

        return torch.ops.higher_order.map(f, xs, y) + xs

    def get_upper_bound_inputs(self):
        return torch.rand(4, 4), torch.rand(4)

    def get_random_inputs(self):
        return torch.rand(2, 4), torch.rand(4)
