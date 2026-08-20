# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import get_args, List, Optional, Sequence, Union

import torch

from torch.utils._pytree import tree_flatten

from typing_extensions import TypeAlias

"""
The data types currently supported for element to be bundled. It should be
consistent with the types in bundled_program.schema.Value.
"""
ConfigValue: TypeAlias = Union[
    torch.Tensor,
    int,
    bool,
    float,
]

"""
The data type of the input for method single execution.
"""
MethodInputType: TypeAlias = Sequence[ConfigValue]

"""
The data type of the output for method single execution.
"""
MethodOutputType: TypeAlias = Sequence[torch.Tensor]

"""
All supported types for input/expected output of MethodTestCase.

Namedtuple is also supported and listed implicitly since it is a subclass of tuple.
"""

# pyre-ignore
DataContainer: TypeAlias = Union[list, tuple, dict]


class MethodTestCase:
    """Test case with inputs and expected outputs
    The expected_outputs are optional and only required if the user wants to verify model outputs after execution.
    """

    def __init__(
        self,
        inputs: MethodInputType,
        expected_outputs: Optional[MethodOutputType] = None,
    ) -> None:
        """Single test case for verifying specific method

        Args:
            inputs: All inputs required by eager_model with specific inference method for one-time execution.

                    It is worth mentioning that, although both bundled program and ET runtime apis support setting input
                    other than `torch.tensor` type, only the input in `torch.tensor` type will be actually updated in
                    the method, and the rest of the inputs will just do a sanity check if they match the default value in method.

            expected_outputs: Expected output of given input for verification. It can be None if user only wants to use the test case for profiling.

        Returns:
            self
        """
        # TODO(gasoonjia): Update type check logic.
        # pyre-ignore [6]: Misalign data type for between MethodTestCase attribute and sanity check.
        self.inputs: List[ConfigValue] = self._flatten_and_sanity_check(inputs)
        self.expected_outputs: List[ConfigValue] = []
        if expected_outputs is not None:
            # pyre-ignore [6]: Misalign data type for between MethodTestCase attribute and sanity check.
            self.expected_outputs = self._flatten_and_sanity_check(expected_outputs)

    def _flatten_and_sanity_check(
        self, unflatten_data: DataContainer
    ) -> List[ConfigValue]:
        """Flat the given data and check its legality

        Args:
            unflatten_data: Data needs to be flatten.

        Returns:
            flatten_data: Flatten data with legal type.
        """

        flatten_data, _ = tree_flatten(unflatten_data)

        for data in flatten_data:
            assert isinstance(
                data,
                get_args(ConfigValue),
            ), "The type of input {} with type {} is not supported.\n".format(
                data, type(data)
            )
            assert not isinstance(
                data,
                type(None),
            ), "The input {} should not be in null type.\n".format(data)

        return flatten_data


@dataclass
class MethodTestSuite:
    """All test info related to verify method

    Attributes:
        method_name: Name of the method to be verified.
        test_cases: All test cases for verifying the method.
    """

    method_name: str
    test_cases: Sequence[MethodTestCase]
