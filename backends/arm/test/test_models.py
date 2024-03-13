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

    @register_test
    class block_conv2d_mean_dim(torch.nn.Module):
        data = rand_test_integers(low=15, high=20, size=(1, 3, 128, 128))
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        permute_memory_to_nhwc = True

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=5, stride=1, bias=False
            )
            # will be specialized to aten.mean.dim
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    rand_test_integers(low=100, high=130, size=(10, 3, 5, 5))
                )

        def forward(self, x):
            x = self.conv2d(x)
            return self.adaptive_avg_pool2d(x)

    # @register_test
    class block_conv_norm_activation(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(1, 3, 256, 256),),
            TosaProfile.MI: (torch.ones(1, 3, 256, 256),),
        }

        permute_memory_to_nhwc = True

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
            )
            self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=False)
            self.relu6 = torch.nn.ReLU6()
            self.eval()

        def forward(self, x):
            x = self.conv2d(x)
            x = self.batch_norm2d(x)
            x = self.relu6(x)
            return x

    @register_test
    class block_bottleneck_residual(torch.nn.Module):
        # This is the essence of MobileNetV2
        # Ref: https://arxiv.org/abs/1801.04381

        inputs = {
            TosaProfile.MI: (torch.ones(1, 64, 81, 81),),
            TosaProfile.BI: (torch.ones(1, 64, 81, 81),),
        }

        permute_memory_to_nhwc = True

        def __init__(self):
            super().__init__()
            # (t, c, n, s) = (6, 96, 1, 1)
            # 1. 1x1 CONV2d + ReLU6 (Pointwise)
            self.pointwise_conv2d = torch.nn.Conv2d(
                in_channels=64, out_channels=384, kernel_size=1, stride=1, groups=1
            )  ## (1, 384, 81, 81)
            self.batch_norm2d_16 = torch.nn.BatchNorm2d(384, affine=False)
            self.relu6 = torch.nn.ReLU6()

            with torch.no_grad():
                self.pointwise_conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(384, 64, 1, 1)))
                    )
                )
                self.pointwise_conv2d.bias.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(384)))
                    )
                )

            # 2. 3x3 DepthwiseConv2d + ReLu6
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=384,
            )  ## (1, 384, H, W)

            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(384, 1, 3, 3)))
                    )
                )
                self.depthwise_conv2d.bias.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(384)))
                    )
                )

            # 3. Linear 1x1 Conv2d
            self.pointwise_conv2d_linear = torch.nn.Conv2d(
                in_channels=384, out_channels=64, kernel_size=1, stride=1, groups=1
            )  ## (1, 64, 81, 81)

            with torch.no_grad():
                self.pointwise_conv2d_linear.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=3, size=(64, 384, 1, 1)))
                    )
                )
                self.pointwise_conv2d_linear.bias.copy_(
                    torch.from_numpy(np.float32(rng.integers(low=1, high=3, size=(64))))
                )

            self.eval()

        def forward(self, x):
            input = x
            # 1x1 CONV2d + ReLU6 (Pointwise)
            x = self.pointwise_conv2d(x)
            x = self.batch_norm2d_16(x)
            x = self.relu6(x)

            # 3x3 DepthwiseConv2d + ReLu6
            x = self.depthwise_conv2d(x)
            x = self.batch_norm2d_16(x)
            x = self.relu6(x)

            # Linear 1x1 Conv2d
            x = self.pointwise_conv2d_linear(x)

            # Final Residual Connection
            x = x + input

            return x
