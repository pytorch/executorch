# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import get_args, List, Union

import torch
from executorch.bundled_program.config import DataContainer

from executorch.bundled_program.tests.common import (
    get_random_config,
    get_random_config_with_eager_model,
    SampleModel,
)
from executorch.extension.pytree import tree_flatten


class TestConfig(unittest.TestCase):
    def assertTensorEqual(self, t1: torch.Tensor, t2: torch.Tensor) -> None:
        self.assertTrue((t1 == t2).all())

    def assertIOListEqual(
        self,
        tl1: List[Union[bool, float, int, torch.Tensor]],
        tl2: List[Union[bool, float, int, torch.Tensor]],
    ) -> None:
        self.assertEqual(len(tl1), len(tl2))
        for t1, t2 in zip(tl1, tl2):
            if type(t1) == torch.Tensor:
                assert type(t1) == type(t2)
                # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                #  `Union[bool, float, int, Tensor]`.
                self.assertTensorEqual(t1, t2)
            else:
                self.assertTrue(t1 == t2)

    def test_create_config(self) -> None:
        n_sets_per_plan_test = 10
        n_execution_plan_tests = 5

        (
            rand_method_names,
            rand_inputs,
            rand_expected_outpus,
            bundled_config,
        ) = get_random_config(
            n_model_inputs=2,
            model_input_sizes=[[2, 2], [2, 2]],
            n_model_outputs=1,
            model_output_sizes=[[2, 2]],
            dtype=torch.int32,
            n_sets_per_plan_test=n_sets_per_plan_test,
            n_execution_plan_tests=n_execution_plan_tests,
        )

        self.assertEqual(
            len(bundled_config.execution_plan_tests), n_execution_plan_tests
        )

        rand_method_names.sort()

        # Compare to see if bundled execution plan test match expectations.
        for plan_test_idx in range(n_execution_plan_tests):
            self.assertEqual(
                bundled_config.execution_plan_tests[plan_test_idx].method_name,
                rand_method_names[plan_test_idx],
            )
            for testset_idx in range(n_sets_per_plan_test):
                self.assertIOListEqual(
                    # pyre-ignore
                    rand_inputs[plan_test_idx][testset_idx],
                    bundled_config.execution_plan_tests[plan_test_idx]
                    .test_sets[testset_idx]
                    .inputs,
                )
                self.assertIOListEqual(
                    # pyre-ignore
                    rand_expected_outpus[plan_test_idx][testset_idx],
                    bundled_config.execution_plan_tests[plan_test_idx]
                    .test_sets[testset_idx]
                    .expected_outputs,
                )

    def test_create_config_from_eager_model(self) -> None:
        n_sets_per_plan_test = 10
        eager_model = SampleModel()
        method_names: List[str] = eager_model.method_names

        rand_inputs, bundled_config = get_random_config_with_eager_model(
            eager_model=eager_model,
            method_names=method_names,
            n_model_inputs=2,
            model_input_sizes=[[2, 2], [2, 2]],
            dtype=torch.int32,
            n_sets_per_plan_test=n_sets_per_plan_test,
        )

        self.assertEqual(len(bundled_config.execution_plan_tests), len(method_names))

        sorted_method_names = sorted(method_names)

        # Compare to see if bundled testcases match expectations.
        for plan_test_idx in range(len(method_names)):
            self.assertEqual(
                bundled_config.execution_plan_tests[plan_test_idx].method_name,
                sorted_method_names[plan_test_idx],
            )
            for testset_idx in range(n_sets_per_plan_test):
                ri = rand_inputs[plan_test_idx][testset_idx]
                self.assertIOListEqual(
                    # pyre-ignore[6]
                    ri,
                    bundled_config.execution_plan_tests[plan_test_idx]
                    .test_sets[testset_idx]
                    .inputs,
                )

                model_outputs = getattr(
                    eager_model, sorted_method_names[plan_test_idx]
                )(*ri)
                if isinstance(model_outputs, get_args(DataContainer)):
                    # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
                    flatten_eager_model_outputs = tree_flatten(model_outputs)
                else:
                    flatten_eager_model_outputs = [
                        model_outputs,
                    ]

                self.assertIOListEqual(
                    flatten_eager_model_outputs,
                    bundled_config.execution_plan_tests[plan_test_idx]
                    .test_sets[testset_idx]
                    .expected_outputs,
                )
