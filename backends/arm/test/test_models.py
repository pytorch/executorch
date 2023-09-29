# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Set of simple models for smoke testing TOSA conversion flow
#

from enum import Enum

import torch

TestList = {}


def register_test(cls):
    TestList[cls.__name__] = cls()
    return cls


# Which TOSA profile to target with a model/inputs
# See https://www.mlplatform.org/tosa/tosa_spec.html#_profiles
class TosaProfile(Enum):
    BI = 0  # Base Inference
    MI = 1  # Main Inference
    MT = 2  # Main Training


class TorchBuilder:
    """The member functions build the PyTorch operators into small networks
    for our tests"""

    def __init__(self):
        pass

    @register_test
    class simple_add(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(5),),
            TosaProfile.MI: (torch.ones(5),),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + x

    @register_test
    class simple_add_broadcast(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(10, 1),
                torch.ones(10, 10),
            ),
            TosaProfile.MI: (
                torch.ones(10, 1),
                torch.ones(10, 10),
            ),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    @register_test
    class simple_linear(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(128, 20),),
            TosaProfile.MI: (torch.ones(128, 20),),
        }

        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(20, 30)
            self.relu6 = torch.nn.ReLU6()

        def forward(self, x):
            x = self.fc(x)
            x = self.relu6(x)
            return x + x

    @register_test
    class simple_conv2d(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(
                    1,
                    3,
                    256,
                    256,
                ),
            ),
            TosaProfile.MI: (torch.ones(1, 3, 256, 256),),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=3, stride=1
            )

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class block_two_conv2d(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(1, 3, 256, 256),),
            TosaProfile.MI: (torch.ones(1, 3, 256, 256),),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=5, stride=1
            )
            self.conv2d_2 = torch.nn.Conv2d(
                in_channels=10, out_channels=15, kernel_size=5, stride=1
            )

        def forward(self, x):
            x = self.conv2d(x)
            x = self.conv2d_2(x)
            return x

    @register_test
    class simple_depthwise_conv2d(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(
                    1,
                    3,
                    256,
                    256,
                ),
            ),
            TosaProfile.MI: (torch.ones(1, 3, 256, 256),),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=3
            )

        def forward(self, x):
            x = self.conv2d(x)
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

        def __init__(self):
            super().__init__()
            self.batch_norm_2d = torch.nn.BatchNorm2d(100, affine=False)
            self.eval()

        def forward(self, x):
            return self.batch_norm_2d(x)

    @register_test
    class simple_avg_pool2d(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (
                torch.ones(
                    20,
                    16,
                    50,
                    32,
                ),
            ),
            TosaProfile.MI: (torch.ones(20, 16, 50, 32),),
        }

        def __init__(self):
            super().__init__()
            self.avg_pool_2d = torch.nn.AvgPool2d(4, stride=2, padding=0)

        def forward(self, x):
            return self.avg_pool_2d(x)

    @register_test
    class simple_softmax(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(2, 3),),
            TosaProfile.MI: (torch.ones(2, 3),),
        }

        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            return self.softmax(x)

    @register_test
    class block_conv_norm_activation(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(1, 3, 256, 256),),
            TosaProfile.MI: (torch.ones(1, 3, 256, 256),),
        }

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
            TosaProfile.BI: (
                torch.ones(
                    1,
                    64,
                    81,
                    81,
                ),
            ),
            TosaProfile.MI: (torch.ones(1, 64, 81, 81),),
        }

        def __init__(self):
            super().__init__()
            # (t, c, n, s) = (6, 96, 1, 1)
            # 1. 1x1 CONV2d + ReLU6 (Pointwise)
            self.pointwise_conv2d = torch.nn.Conv2d(
                in_channels=64, out_channels=384, kernel_size=1, stride=1, groups=1
            )  ## (1, 384, 81, 81)
            self.batch_norm2d_16 = torch.nn.BatchNorm2d(384, affine=False)
            self.relu6 = torch.nn.ReLU6()

            # 2. 3x3 DepthwiseConv2d + ReLu6
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=384,
            )  ## (1, 384, H, W)

            # 3. Linear 1x1 Conv2d
            self.pointwise_conv2d_linear = torch.nn.Conv2d(
                in_channels=384, out_channels=64, kernel_size=1, stride=1, groups=1
            )  ## (1, 64, 81, 81)

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
