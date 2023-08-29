#
# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause
#

#
# Set of simple models for smoke testing TOSA conversion flow
#

import torch
from enum import Enum

TestList = {}
def register_test( cls ):
    TestList[cls.__name__] = cls()
    return cls

# Which TOSA profile to target with a model/inputs
# See https://www.mlplatform.org/tosa/tosa_spec.html#_profiles
class TosaProfile(Enum):
    BI = 0 # Base Inference
    MI = 1 # Main Inference
    MT = 2 # Main Training

class TorchBuilder:
    """The member functions build the PyTorch operators into small networks
    for our tests"""
    def __init__(self):
        pass

    @register_test
    class simple_add(torch.nn.Module):
        inputs = {
            TosaProfile.BI: ( torch.ones(5, dtype=torch.int32), ),
            TosaProfile.MI: ( torch.ones(5), ),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x+x

    @register_test
    class simple_add_broadcast(torch.nn.Module):
        inputs = {
            TosaProfile.BI: ( torch.ones(10,1, dtype=torch.int32), torch.ones(10,10, dtype=torch.int32), ),
            TosaProfile.MI: ( torch.ones(10,1), torch.ones(10,10), ),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x+y

    @register_test
    class simple_linear(torch.nn.Module):
        inputs = {
            #TODO: RuntimeError: mat1 and mat2 must have the same dtype, but got Int and Float
            #TosaProfile.BI: ( torch.ones(128,20, dtype=torch.int32), ),
            TosaProfile.MI: ( torch.ones(128,20), ),
        }

        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear( 20, 30 )
            self.relu6 = torch.nn.ReLU6()

        def forward(self, x):
            x = self.fc( x )
            x = self.relu6(x)
            return x+x

    @register_test
    class simple_conv2d(torch.nn.Module):
        inputs = {
            #TODO: fails input char, bias float
            #TosaProfile.BI: ( torch.ones(1,3,256,256, dtype=torch.int8), ),
            TosaProfile.MI: ( torch.ones(1,3,256,256), ),
        }

        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)

        def forward(self, x):
            x = self.conv2d(x)
            return x

    @register_test
    class simple_div(torch.nn.Module):
        inputs = {
            # TODO: BUG: need to codegen for integer div, current float/recip one is not valid BI
            #TosaProfile.BI: ( torch.ones(5, dtype=torch.int8), torch.ones(5, dtype=torch.int8), ),
            TosaProfile.MI: ( torch.ones(5), torch.ones(5), ),
        }

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.div(x, y)

    @register_test
    class simple_batch_norm(torch.nn.Module):
        inputs = {
            # "RuntimeError: "batch_norm" not implemented for 'Char'"
            # TosaProfile.BI: ( torch.ones(20,100,35,45, dtype=torch.int8), ),
            TosaProfile.MI: ( torch.ones(20,100,35,45), ),
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
            # TosaProfile.BI: ( torch.ones(20, 16, 50, 32, dtype=torch.int8), ),
            TosaProfile.MI: ( torch.ones(20, 16, 50, 32), ),
        }

        def __init__(self):
            super().__init__()
            self.avg_pool_2d = torch.nn.AvgPool2d(4, stride=2, padding=0)

        def forward(self, x):
            return self.avg_pool_2d(x)
