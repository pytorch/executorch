# pyre-strict

from dataclasses import dataclass
from typing import Any, Dict, get_args, List, Union

import torch
from executorch.bundled_program.schema import (
    BundledAttachment,
    BundledAttachmentValue,
    BundledBool,
    BundledBytes,
    BundledDouble,
    BundledInt,
    BundledString,
)
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


"""
All supported types for bundled attachment value.
"""
PrimTypeForAttachment: TypeAlias = Union[bool, bytes, int, float, str]


@dataclass
class ConfigIOSet:
    """Type of data BundledConfig stored for each validation set."""

    inputs: List[ConfigValue]
    expected_outputs: List[ConfigValue]


@dataclass
class ConfigExecutionPlanTest:
    """All info related to verify execution plan"""

    test_sets: List[ConfigIOSet]
    metadata: List[BundledAttachment]


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
        # pyre-ignore
        metadatas: List[Dict] = None,
        # pyre-ignore
        **kwargs,
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

            metadatas: Other info needs to specific verify each execution plan. Its length should
                        be the same as the number of execution plans. Each dict in the list should
                        be for the execution plan with same index. The key of each element
                        should be in string. For the value, we now support multiple common types
                        (bool, int, float, str) plus bytes. For other types, user should manually
                        convert them to supported types (recommend bytes).

                        Same as kwargs below.

            **kwargs: Other info need to be attached. All given **kwargs will be
                      transformed into attachment and saved into BundledConfig.
        """
        BundledConfig._check_io_type(inputs)
        BundledConfig._check_io_type(expected_outputs)
        assert len(inputs) == len(expected_outputs), (
            "length of inputs and expected_outputs should match,"
            + " but got {} and {}".format(len(inputs), len(expected_outputs))
        )

        if metadatas is None:
            metadatas = [{} for _ in range(len(expected_outputs))]
        for metadata_plan in metadatas:
            BundledConfig._check_attachemnt_type(metadata_plan)
        assert len(inputs) == len(
            metadatas
        ), "length of I/O and meta data should match," + " but got {} and {}".format(
            len(inputs), len(metadatas)
        )

        BundledConfig._check_attachemnt_type(kwargs)

        self.execution_plan_tests: List[
            ConfigExecutionPlanTest
        ] = BundledConfig._gen_execution_plan_tests(inputs, expected_outputs, metadatas)

        self.attachments: List[
            BundledAttachment
        ] = BundledConfig._gen_bundled_attachment(kwargs)

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
    # pyre-ignore
    def _check_attachemnt_type(attachment):
        """Check the type of each attachment. Each attachment should be in Dict[str, Any]"""

        assert type(attachment) is dict
        for k in attachment:
            assert (
                type(k) is str
            ), "key of attachment should be in str, but found {}".format(type(k))

    @staticmethod
    def _gen_execution_plan_tests(
        # pyre-ignore
        inputs: List[List[Any]],
        # pyre-ignore
        expected_outputs: List[List[Any]],
        # pyre-ignore
        metadatas: List[Dict] = None,
    ) -> List[ConfigExecutionPlanTest]:
        """Generate execution plan test given inputs, expected outputs and metadatas for verifying each execution plan"""

        execution_plan_tests: List[ConfigExecutionPlanTest] = []

        for (
            inputs_per_plan_test,
            expect_outputs_per_plan_test,
            metadata_per_plan_test,
        ) in zip(inputs, expected_outputs, metadatas):
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

            # transfer meta data into BundledAttachment for each execution plan
            metadata: List[BundledAttachment] = BundledConfig._gen_bundled_attachment(
                metadata_per_plan_test
            )

            execution_plan_tests.append(
                ConfigExecutionPlanTest(
                    test_sets=test_sets,
                    metadata=metadata,
                )
            )
        return execution_plan_tests

    @staticmethod
    def convert_prim_val_to_attachement_val(
        prim_val: PrimTypeForAttachment,
    ) -> BundledAttachmentValue:
        """Convert primitive value to bundled attachment value"""
        if type(prim_val) is int:
            v = BundledAttachmentValue(val=BundledInt(int_val=prim_val))
        elif type(prim_val) is float:
            v = BundledAttachmentValue(val=BundledDouble(double_val=prim_val))
        elif type(prim_val) is bool:
            v = BundledAttachmentValue(val=BundledBool(bool_val=prim_val))
        elif type(prim_val) is str:
            v = BundledAttachmentValue(val=BundledString(string_value=prim_val))
        elif type(prim_val) is bytes:
            v = BundledAttachmentValue(val=BundledBytes(bytes_value=prim_val))
        else:
            raise AssertionError(
                "Bundled value should be one of the following types: string, float, int, bool, bytes,"
                + "but got {}".format(type(prim_val))
            )
        return v

    @staticmethod
    def _gen_bundled_attachment(
        attached_dict: Dict[str, Any]
    ) -> List[BundledAttachment]:
        """Generate bundle attachment from given dictionary"""

        bundled_attachment: List[BundledAttachment] = []
        for attached_key, attached_val in attached_dict.items():
            bundled_attachment.append(
                BundledAttachment(
                    key=attached_key,
                    val=BundledConfig.convert_prim_val_to_attachement_val(attached_val),
                )
            )

        return bundled_attachment
