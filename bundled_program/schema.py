# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List, Union

from executorch.exir.scalar_type import ScalarType


@dataclass
class Tensor:
    """All information we need to bundle for a tensor EValue input."""

    # The scalar type of Tensor
    scalar_type: ScalarType
    # The target sizes of the tensor.
    sizes: List[int]
    # The contents of the corresponding tensor.
    data: bytes
    dim_order: List[bytes]


@dataclass
class Int:
    int_val: int


@dataclass
class Bool:
    bool_val: bool


@dataclass
class Double:
    double_val: float


ValueUnion = Union[
    Tensor,
    Int,
    Double,
    Bool,
]


@dataclass
class Value:
    """Abstraction for BundledIOSet values"""

    val: "ValueUnion"


@dataclass
class BundledIOSet:
    """All inputs and referenced outputs needs for single verification."""

    # All inputs required by Program for execution. Its length should be
    # equal to the length of program inputs.
    inputs: List[Value]

    # The expected outputs generated while running the model in eager mode
    # using the inputs provided. Its length should be equal to the length
    # of program outputs.
    expected_outputs: List[Value]


@dataclass
class BundledExecutionPlanTest:
    """Context for testing and verifying an exceution plan."""

    # The name of the method to test; e.g., "forward" for the forward() method
    # of an nn.Module. This name match a method defined by the ExecuTorch
    # program.
    method_name: str

    # Sets of input/outputs to test with.
    test_sets: List[BundledIOSet]


@dataclass
class BundledProgram:
    """ExecuTorch program bunlded with data for verification."""

    # Schema version.
    version: int

    # Test sets and other meta datas to verify the whole program.
    # Each BundledExecutionPlanTest should be used for the execution plan of program sharing same index.
    # Its length should be equal to the number of execution plans in program.
    execution_plan_tests: List[BundledExecutionPlanTest]

    # The binary data of a serialized ExecuTorch program.
    program: bytes
