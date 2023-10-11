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
The data type of the input for method single execution.
"""
MethodInputType: TypeAlias = List[ConfigValue]

"""
The data type of the output for method single execution.
"""
MethodOutputType: TypeAlias = List[torch.Tensor]

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

    method_name: str
    test_sets: List[ConfigIOSet]


class BundledConfig:
    """All information needed to verify a model.

    Public Attributes:
        execution_plan_tests: inputs, expected outputs, and other info for each execution plan verification.
        attachments: Other info need to be attached.
    """

    def __init__(
        self,
        method_names: List[str],
        inputs: List[List[MethodInputType]],
        expected_outputs: List[List[MethodOutputType]],
    ) -> None:
        """Contruct the config given inputs and expected outputs

        Args:
            method_names: All method names need to be verified in program.
            inputs: All sets of input need to be test on for all methods. Each list
                    of `inputs` is all sets which will be run on the method in the
                    program with corresponding method name. Each set of any `inputs` element should
                    contain all inputs required by eager_model with the same inference function
                    as corresponding execution plan for one-time execution.

                    It is worth mentioning that, although both bundled program and ET runtime apis support setting input
                    other than torch.tensor type, only the input in torch.tensor type will be actually updated in
                    the method, and the rest of the inputs will just do a sanity check if they match the default value in method.

            expected_outputs: Expected outputs for inputs sharing same index. The size of
                    expected_outputs should be the same as the size of inputs and provided method_names.

        Returns:
            self
        """
        BundledConfig._check_io_type(inputs)
        BundledConfig._check_io_type(expected_outputs)

        for m_name in method_names:
            assert isinstance(m_name, str)

        assert len(method_names) == len(inputs) == len(expected_outputs), (
            "length of method_names, inputs and expected_outputs should match,"
            + " but got {}, {} and {}".format(
                len(method_names), len(inputs), len(expected_outputs)
            )
        )

        self.execution_plan_tests: List[
            ConfigExecutionPlanTest
        ] = BundledConfig._gen_execution_plan_tests(
            method_names, inputs, expected_outputs
        )

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
        method_names: List[str],
        inputs: List[List[MethodInputType]],
        expected_outputs: List[List[MethodOutputType]],
    ) -> List[ConfigExecutionPlanTest]:
        """Generate execution plan test given inputs, expected outputs for verifying each execution plan"""

        execution_plan_tests: List[ConfigExecutionPlanTest] = []

        for (
            m_name,
            inputs_per_plan_test,
            expect_outputs_per_plan_test,
        ) in zip(method_names, inputs, expected_outputs):
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
                    method_name=m_name,
                    test_sets=test_sets,
                )
            )

        # sort the execution plan tests by method name to in line with core program emitter.
        execution_plan_tests.sort(key=lambda x: x.method_name)

        return execution_plan_tests
