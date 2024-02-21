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

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x.view(2, 5)
            return x

    @register_test
    class simple_add(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.ones(5),),
            TosaProfile.MI: (torch.ones(5),),
            TosaProfile.BI_INT: (torch.ones(5, dtype=torch.int32),),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + x

    @register_test
    class simple_add_2(torch.nn.Module):
        inputs = {
            TosaProfile.BI_INT: (
                torch.ones(5, dtype=torch.int32),
                torch.ones(5, dtype=torch.int32),
            ),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

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

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    @register_test
    class simple_linear(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.rand(1, 2),),
            TosaProfile.MI: (torch.rand(1, 2),),
        }

        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.fc = torch.nn.Linear(2, 3)

        def forward(self, x):
            x = self.fc(x)
            return x

    @register_test
    class simple_linear_rank4(torch.nn.Module):
        inputs = {
            TosaProfile.BI: (torch.rand(5, 10, 25, 20),),
            TosaProfile.MI: (torch.rand(5, 10, 25, 20),),
        }

        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
            self.fc = torch.nn.Linear(20, 30)

        def forward(self, x):
            x = self.fc(x)
            return x

    """Currenly we compare the quantized result directly with the floating point result, to avoid a noticable
       precision difference due to wide random numerical distribution, generate small random value range for
       convolution testing instead for now"""

    @register_test
    class simple_conv2d_2x2_3x1x40x40_non_bias(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(3, 1, 40, 40)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=2, stride=1, bias=False
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=10, size=(1, 1, 2, 2)))
                    )
                )

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class simple_conv2d_3x3_1x3x256x256_stride1(torch.nn.Module):
        data = torch.ones(1, 3, 256, 256)
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=3, stride=1
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(10, 3, 3, 3)))
                    )
                )
                self.conv2d.bias.copy_(
                    torch.from_numpy(np.float32(rng.integers(low=1, high=4, size=(10))))
                )

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class simple_conv2d_1x1_1x2x128x128_stride1(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(1, 2, 128, 128)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=1, stride=1
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(1, 2, 1, 1)))
                    )
                )
                self.conv2d.bias.copy_(
                    torch.from_numpy(np.float32(rng.integers(low=1, high=4, size=(1))))
                )

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class simple_conv2d_2x2_1x1x14x14_stride2(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(1, 1, 14, 14)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=2, stride=2
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=4, size=(1, 1, 2, 2)))
                    )
                )
                self.conv2d.bias.copy_(
                    torch.from_numpy(np.float32(rng.integers(low=1, high=4, size=(1))))
                )

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class simple_conv2d_5x5_3x2x128x128_stride1(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(3, 2, 128, 128)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=2, out_channels=3, kernel_size=5, stride=1
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(
                    torch.from_numpy(
                        np.float32(rng.integers(low=1, high=10, size=(1, 1, 5, 5)))
                    )
                )
                self.conv2d.bias.copy_(torch.ones(3, dtype=torch.float))

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class block_two_conv2d_non_bias(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(1, 3, 256, 256)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=5, stride=1, bias=False
            )
            self.conv2d_2 = torch.nn.Conv2d(
                in_channels=10, out_channels=15, kernel_size=5, stride=1, bias=False
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(torch.ones(10, 3, 5, 5, dtype=torch.float))
                self.conv2d_2.weight.copy_(torch.ones(15, 10, 5, 5, dtype=torch.float))

        def forward(self, x):
            x = self.conv2d(x)
            x = self.conv2d_2(x)
            return x

    @register_test
    class block_two_conv2d(torch.nn.Module):
        data = torch.from_numpy(
            np.float32(rng.integers(low=10, high=20, size=(1, 3, 256, 256)))
        )
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=10, kernel_size=5, stride=1
            )
            self.conv2d_2 = torch.nn.Conv2d(
                in_channels=10, out_channels=15, kernel_size=5, stride=1
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(torch.ones(10, 3, 5, 5, dtype=torch.float))
                self.conv2d.bias.copy_(torch.ones(10))
                self.conv2d_2.weight.copy_(torch.ones(15, 10, 5, 5, dtype=torch.float))
                self.conv2d_2.bias.copy_(torch.ones(15))

        def forward(self, x):
            x = self.conv2d(x)
            x = self.conv2d_2(x)
            return x

    @register_test
    class simple_depthwise_conv2d_3x3x3_1x3x256x256_group3_stride1(torch.nn.Module):
        data = torch.ones(1, 3, 256, 256)
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            """ in_channels == out_channels """
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=3
            )
            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    rand_test_integers(low=100, high=130, size=(3, 1, 3, 3))
                )
                self.depthwise_conv2d.bias.copy_(torch.ones(3))

        def forward(self, x):
            x = self.depthwise_conv2d(x)
            return x

    @register_test
    class simple_depthwise_conv2d_8x4x3x3_1x4x256x256_group4_stride1(torch.nn.Module):
        data = torch.ones(1, 4, 256, 256)
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            """ in_channels * k == out_channels, where k = 2 """
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=3, stride=1, groups=4
            )
            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    rand_test_integers(low=21, high=25, size=(8, 1, 3, 3))
                )
                self.depthwise_conv2d.bias.copy_(
                    rand_test_integers(low=1, high=4, size=(8))
                )

        def forward(self, x):
            x = self.depthwise_conv2d(x)
            return x

    @register_test
    class simple_depthwise_conv2d_8x16x3_2x8x198x198_group8_stride3(torch.nn.Module):
        data = torch.ones(2, 8, 198, 198)
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=3, groups=8
            )
            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    rand_test_integers(low=80, high=110, size=(16, 1, 3, 3))
                )
                self.depthwise_conv2d.bias.copy_(
                    rand_test_integers(low=1, high=4, size=(16))
                )

        def forward(self, x):
            x = self.depthwise_conv2d(x)
            return x

    @register_test
    class simple_depthwise_conv2d_3x3_1x4x256x256_group4_non_bias(torch.nn.Module):
        data = torch.ones(1, 4, 256, 256)
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=1,
                groups=4,
                bias=False,
            )
            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    rand_test_integers(low=30, high=50, size=(8, 1, 3, 3))
                )

        def forward(self, x):
            x = self.depthwise_conv2d(x)
            return x

    @register_test
    class block_two_depthwise_conv2d(torch.nn.Module):
        data = rand_test_integers(low=10, high=20, size=(2, 4, 64, 64))

        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            self.depthwise_conv2d = torch.nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=3, stride=1, groups=4
            )
            self.depthwise_conv2d_2 = torch.nn.Conv2d(
                in_channels=8, out_channels=24, kernel_size=3, stride=1, groups=8
            )
            with torch.no_grad():
                self.depthwise_conv2d.weight.copy_(
                    rand_test_integers(low=11, high=14, size=(8, 1, 3, 3))
                )
                self.depthwise_conv2d.bias.copy_(torch.ones(8))
                self.depthwise_conv2d_2.weight.copy_(
                    rand_test_integers(low=11, high=14, size=(24, 1, 3, 3))
                )
                self.depthwise_conv2d.bias.copy_(torch.ones(8))

        def forward(self, x):
            x = self.depthwise_conv2d(x)
            x = self.depthwise_conv2d_2(x)
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
    class simple_mean_dim(torch.nn.Module):
        data = rand_test_integers(low=15, high=20, size=(20, 16, 50, 32))
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

        def __init__(self):
            super().__init__()
            # will be specialized to aten.mean.dim
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return self.adaptive_avg_pool2d(x)

    @register_test
    class block_conv2d_mean_dim(torch.nn.Module):
        data = rand_test_integers(low=15, high=20, size=(1, 3, 128, 128))
        inputs = {
            TosaProfile.BI: (data,),
            TosaProfile.MI: (data,),
        }

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

    # @register_test
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

    # @register_test
    class block_bottleneck_residual(torch.nn.Module):
        # This is the essence of MobileNetV2
        # Ref: https://arxiv.org/abs/1801.04381

        inputs = {
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
