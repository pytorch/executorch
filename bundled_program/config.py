# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Any, get_args, List, Union

import torch
from executorch.extension.pytree import tree_flatten

from typing_extensions import TypeAlias

"""
The data types currently supported for element to be bundled. It should be
consistent with the types in bundled_program.schema.BundledValue.
"""
ConfigValue: TypeAlias = Union[
    torch.Tensor,
    int,
    bool,
    float,
]

"""
All supported types for input/expected output of test set.

Namedtuple is also supported and listed implicity since it is a subclass of tuple.
"""

# pyre-ignore
DataContainer: TypeAlias = Union[list, tuple, dict]


@dataclass
class ConfigIOSet:
    """Type of data BundledConfig stored for each validation set."""

    inputs: List[ConfigValue]
    expected_outputs: List[ConfigValue]


@dataclass
class ConfigExecutionPlanTest:
    """All info related to verify execution plan"""

    test_sets: List[ConfigIOSet]


class BundledConfig:
    """All information needed to verify a model.

    Public Attributes:
        execution_plan_tests: inputs, expected outputs, and other info for each execution plan verification.
        attachments: Other info need to be attached.
    """

    def __init__(
        self,
        # pyre-ignore
        inputs: List[List[Any]],
        # pyre-ignore
        expected_outputs: List[List[Any]],
    ) -> None:
        """Contruct the config given inputs and expected outputs

        Args:
            inputs: All sets of input need to be test on for all execution plans. Each list
                    of `inputs` is all sets which will be run on the execution plan in the
                    program sharing same index. Each set of any `inputs` element should
                    contain all inputs required by eager_model with the same inference function
                    as corresponding execution plan for one-time execution.

                    Please note that currently we do not have any consensus about the mapping rule
                    between inference name in eager_model and execution plan id in executorch
                    program. Hence, user should take care of the data order in `inputs`: each list
                    of `inputs` is all sets which will be run on the execution plan with same index,
                    not the inference function with same index in the result of get_inference_name.
                    Same as the `expected_outputs` and `metadatas` below.

                    It shouldn't be a problem if there's only one inferenece function per model.

            expected_outputs: Expected outputs for inputs sharing same index. The size of
                    expected_outputs should be the same as the size of inputs.
        """
        BundledConfig._check_io_type(inputs)
        BundledConfig._check_io_type(expected_outputs)
        assert len(inputs) == len(expected_outputs), (
            "length of inputs and expected_outputs should match,"
            + " but got {} and {}".format(len(inputs), len(expected_outputs))
        )

        self.execution_plan_tests: List[
            ConfigExecutionPlanTest
        ] = BundledConfig._gen_execution_plan_tests(inputs, expected_outputs)

    @staticmethod
    # TODO(T138930448): Give pyre-ignore commands appropriate warning type and comments.
    # pyre-ignore
    def _tree_flatten(unflatten_data: Any) -> List[ConfigValue]:
        """Flat the given data and check its legality

        Args:
            unflatten_data: Data needs to be flatten.

        Returns:
            flatten_data: Flatten data with legal type.
        """
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

    @staticmethod
    # pyre-ignore
    def _check_io_type(test_data_program: List[List[Any]]) -> None:
        """Check the type of each set of inputs or exepcted_outputs

        Each test set of inputs or expected_outputs will be put into the config
        should be one of the sub-type in DataContainer.

        Args:
            test_data_program: inputs or expected_outputs to be put into the config
                               to verify the whole program.

        """
        for test_data_execution_plan in test_data_program:
            for test_set in test_data_execution_plan:
                assert isinstance(test_set, get_args(DataContainer))

    @staticmethod
    def _gen_execution_plan_tests(
        # pyre-ignore
        inputs: List[List[Any]],
        # pyre-ignore
        expected_outputs: List[List[Any]],
    ) -> List[ConfigExecutionPlanTest]:
        """Generate execution plan test given inputs, expected outputs for verifying each execution plan"""

        execution_plan_tests: List[ConfigExecutionPlanTest] = []

        for (
            inputs_per_plan_test,
            expect_outputs_per_plan_test,
        ) in zip(inputs, expected_outputs):
            test_sets: List[ConfigIOSet] = []

            # transfer I/O sets into ConfigIOSet for each execution plan
            assert len(inputs_per_plan_test) == len(expect_outputs_per_plan_test), (
                "The number of input and expected output for identical execution plan should be the same,"
                + " but got {} and {}".format(
                    len(inputs_per_plan_test), len(expect_outputs_per_plan_test)
                )
            )
            for unflatten_input, unflatten_expected_output in zip(
                inputs_per_plan_test, expect_outputs_per_plan_test
            ):
                flatten_inputs = BundledConfig._tree_flatten(unflatten_input)
                flatten_expected_output = BundledConfig._tree_flatten(
                    unflatten_expected_output
                )
                test_sets.append(
                    ConfigIOSet(
                        inputs=flatten_inputs, expected_outputs=flatten_expected_output
                    )
                )

            execution_plan_tests.append(
                ConfigExecutionPlanTest(
                    test_sets=test_sets,
                )
            )
        return execution_plan_tests
