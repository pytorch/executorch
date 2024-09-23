# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.fx import Node

OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = "executorch.backends.arm.quantizer.arm_quantizer_utils"


class OperatorConfig(NamedTuple):
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    config: QuantizationConfig
    operators: List[OperatorPatternType]


AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        QuantizationConfig,
        Optional[Callable[[Node], bool]],
    ],
    Optional[List[List[Node]]],
]
OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}


def register_annotator(op: str):
    def decorator(annotator: AnnotatorType):
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


from . import (  # noqa
    adaptive_ang_pool2d_annotator,
    add_annotator,
    cat_annotator,
    conv_annotator,
    linear_annotator,
    max_pool2d_annotator,
    mm_annotator,
    mul_annotator,
    one_to_one_annotator,
    sigmoid_annotator,
    sub_annotator,
)
