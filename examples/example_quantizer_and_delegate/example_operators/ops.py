# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from executorch.examples.example_quantizer_and_delegate.example_operators.adaptive_avg_pool2d import (
    AdaptiveAvgPool2dNode,
)

from executorch.examples.example_quantizer_and_delegate.example_operators.add import (
    AddNode,
)
from executorch.examples.example_quantizer_and_delegate.example_operators.conv2d import (
    Conv2DNode,
)
from executorch.examples.example_quantizer_and_delegate.example_operators.conv_relu import (
    ConvReluNode,
)
from executorch.examples.example_quantizer_and_delegate.example_operators.dropout import (
    DropOutNode,
)
from executorch.examples.example_quantizer_and_delegate.example_operators.flatten import (
    FlattenNode,
)
from executorch.examples.example_quantizer_and_delegate.example_operators.linear import (
    LinearNode,
)

# The ordering of this is important as the quantizer will try to match the patterns in this order.
# That's why we want to match the fused patterns first and then the non-fused ones.
module_to_annotator = OrderedDict(
    {
        ConvReluNode().pattern: ConvReluNode(),
        Conv2DNode().pattern: Conv2DNode(),
        LinearNode().pattern: LinearNode(),
        AddNode().pattern: AddNode(),
        AdaptiveAvgPool2dNode().pattern: AdaptiveAvgPool2dNode(),
        FlattenNode().pattern: FlattenNode(),
        DropOutNode().pattern: DropOutNode(),
    }
)
