# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import get_args, List, Optional, Sequence, Union

import torch
from executorch.extension.pytree import tree_flatten

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
All supported types for input/expected output of MethodTestCase.

Namedtuple is also supported and listed implicity since it is a subclass of tuple.
"""

# pyre-ignore
DataContainer: TypeAlias = Union[list, tuple, dict]


class MethodTestCase:
    """Test case with inputs and expected outputs
    The expected_outputs could be None if user only want to user the test case for profiling."""

    def __init__(
        self, inputs: DataContainer, expected_outputs: Optional[DataContainer] = None
    ) -> None:
        self.inputs: List[ConfigValue] = self._flatten_and_sanity_check(inputs)
        self.expected_outputs: List[ConfigValue] = []
        if expected_outputs:
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

        assert isinstance(
            unflatten_data, get_args(DataContainer)
        ), f"The input or expected output of MethodTestCase should be in list, tuple or dict, but got {type(unflatten_data)}."

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
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
    """All info related to verify method"""

    method_name: str
    test_cases: Sequence[MethodTestCase]
