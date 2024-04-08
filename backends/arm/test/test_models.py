# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Set of simple models for smoke testing TOSA conversion flow
#

from enum import Enum

import numpy as np

import torch


TestList = {}

# Seed the RNG a convenient number so that we get the same random tests for each test each time
seed = 42
rng = np.random.default_rng(seed)


def register_test(cls):
    TestList[cls.__name__] = cls()
    return cls


# Which TOSA profile to target with a model/inputs
# See https://www.mlplatform.org/tosa/tosa_spec.html#_profiles
class TosaProfile(Enum):
    BI = 0  # Base Inference
    MI = 1  # Main Inference
    MT = 2  # Main Training
    BI_INT = 3  # integer only BI subset tests (for test graphs)


def rand_test_integers(low, high, size):
    return torch.from_numpy(np.float32(rng.integers(low, high, size)))


class TorchBuilder:
    """The member functions build the PyTorch operators into small networks
    for our tests"""

    def __init__(self):
        pass

    @register_test
    class simple_clone(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(10),),
            TosaProfile.MI: (torch.ones(10),),
        }

        permute_memory_to_nhwc = False

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x.clone()
            return x

    @register_test
    class simple_view(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(10),),
            TosaProfile.MI: (torch.ones(10),),
        }

        permute_memory_to_nhwc = False

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x.view(2, 5)
            return x

    @register_test
    class simple_add_broadcast(torch.nn.Module):
        inputs = {
            TosaProfile.BI_INT: (
                torch.ones(10, 1, dtype=torch.int32),
                torch.ones(10, 10, dtype=torch.int32),
            ),
            TosaProfile.BI: (
                torch.ones(10, 1),
                torch.ones(10, 10),
            ),
            TosaProfile.MI: (
                torch.ones(10, 1),
                torch.ones(10, 10),
            ),
        }

        permute_memory_to_nhwc = False

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

            return x

    @register_test
    class simple_div(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(
                    5,
                ),
                torch.ones(
                    5,
                ),
            ),
            TosaProfile.MI: (
                torch.ones(5),
                torch.ones(5),
            ),
        }

        permute_memory_to_nhwc = False

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.div(x, y)

    @register_test
    class simple_batch_norm(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(
                    20,
                    100,
                    35,
                    45,
                ),
            ),
            TosaProfile.MI: (torch.ones(20, 100, 35, 45),),
        }

        permute_memory_to_nhwc = False

        def __init__(self):
            super().__init__()
            self.batch_norm_2d = torch.nn.BatchNorm2d(100, affine=False)
            self.eval()

        def forward(self, x):
            return self.batch_norm_2d(x)
