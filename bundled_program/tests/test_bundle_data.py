# pyre-strict

import unittest
from typing import List

import torch
from executorch.bundled_program.config import ConfigValue
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.schema import (
    BundledAttachment,
    BundledBool,
    BundledDouble,
    BundledInt,
    BundledTensor,
    BundledValue,
)
from executorch.bundled_program.tests.common import get_common_program
from executorch.exir.serialize import serialize_to_flatbuffer


class TestBundle(unittest.TestCase):
    def assertIOsetDataEqual(
        self,
        program_ioset_data: List[BundledValue],
        config_ioset_data: List[ConfigValue],
    ) -> None:
        self.assertEqual(len(program_ioset_data), len(config_ioset_data))
        for program_element, config_element in zip(
            program_ioset_data, config_ioset_data
        ):
            if isinstance(program_element.val, BundledTensor):
                # TODO: Update to check the bundled input share the same type with the config input after supporting multiple types.
                self.assertTrue(isinstance(config_element, torch.Tensor))
                self.assertEqual(program_element.val.sizes, list(config_element.size()))
                # TODO(gasoonjia): Check the inner data.
            elif type(program_element.val) == BundledInt:
                self.assertEqual(program_element.val.int_val, config_element)
            elif type(program_element.val) == BundledDouble:
                self.assertEqual(program_element.val.double_val, config_element)
            elif type(program_element.val) == BundledBool:
                self.assertEqual(program_element.val.bool_val, config_element)

    def assertAttachmentEqual(
        self,
        config_attachments: List[BundledAttachment],
        bundled_attachments: List[BundledAttachment],
    ) -> None:
        self.assertEqual(len(config_attachments), len(bundled_attachments))
        for config_attachment, bundled_attachment in zip(
            config_attachments, bundled_attachments
        ):
            self.assertEqual(config_attachment.key, bundled_attachment.key)
            self.assertEqual(config_attachment.val, bundled_attachment.val)

    def test_bundled_program(self) -> None:
        program, bundled_config = get_common_program()

        bundled_program = create_bundled_program(program, bundled_config)

        for plan_id in range(len(program.execution_plan)):
            bundled_plan_test = bundled_program.execution_plan_tests[plan_id]
            config_plan_test = bundled_config.execution_plan_tests[plan_id]
            self.assertEqual(
                len(bundled_plan_test.test_sets), len(config_plan_test.test_sets)
            )
            for bundled_program_ioset, bundled_config_ioset in zip(
                bundled_plan_test.test_sets, config_plan_test.test_sets
            ):
                self.assertIOsetDataEqual(
                    bundled_program_ioset.inputs, bundled_config_ioset.inputs
                )
                self.assertIOsetDataEqual(
                    bundled_program_ioset.expected_outputs,
                    bundled_config_ioset.expected_outputs,
                )
            self.assertAttachmentEqual(
                bundled_plan_test.metadata, config_plan_test.metadata
            )

        self.assertEqual(bundled_program.program, serialize_to_flatbuffer(program))
        self.assertAttachmentEqual(
            bundled_program.attachments, bundled_config.attachments
        )
