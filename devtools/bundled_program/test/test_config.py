# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import get_args, List, Union

import torch
from executorch.devtools.bundled_program.config import DataContainer

from executorch.devtools.bundled_program.util.test_util import (
    get_random_test_suites,
    get_random_test_suites_with_eager_model,
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
            if isinstance(t1, torch.Tensor):
                assert isinstance(t2, torch.Tensor)
                self.assertTensorEqual(t1, t2)
            else:
                self.assertTrue(t1 == t2)

    def test_create_test_suites(self) -> None:
        n_sets_per_plan_test = 10
        n_method_test_suites = 5

        (
            rand_method_names,
            rand_inputs,
            rand_expected_outpus,
            method_test_suites,
        ) = get_random_test_suites(
            n_model_inputs=2,
            model_input_sizes=[[2, 2], [2, 2]],
            n_model_outputs=1,
            model_output_sizes=[[2, 2]],
            dtype=torch.int32,
            n_sets_per_plan_test=n_sets_per_plan_test,
            n_method_test_suites=n_method_test_suites,
        )

        self.assertEqual(len(method_test_suites), n_method_test_suites)

        # Compare to see if bundled execution plan test match expectations.
        for method_test_suite_idx in range(n_method_test_suites):
            self.assertEqual(
                method_test_suites[method_test_suite_idx].method_name,
                rand_method_names[method_test_suite_idx],
            )
            for testset_idx in range(n_sets_per_plan_test):
                self.assertIOListEqual(
                    # pyre-ignore [6]: expected `List[Union[bool, float, int, Tensor]]` but got `Sequence[Union[bool, float, int, Tensor]]
                    rand_inputs[method_test_suite_idx][testset_idx],
                    method_test_suites[method_test_suite_idx]
                    .test_cases[testset_idx]
                    .inputs,
                )
                self.assertIOListEqual(
                    # pyre-ignore [6]: expected `List[Union[bool, float, int, Tensor]]` but got `Sequence[Union[bool, float, int, Tensor]]
                    rand_expected_outpus[method_test_suite_idx][testset_idx],
                    method_test_suites[method_test_suite_idx]
                    .test_cases[testset_idx]
                    .expected_outputs,
                )

    def test_create_test_suites_from_eager_model(self) -> None:
        n_sets_per_plan_test = 10
        eager_model = SampleModel()
        method_names: List[str] = eager_model.method_names

        rand_inputs, method_test_suites = get_random_test_suites_with_eager_model(
            eager_model=eager_model,
            method_names=method_names,
            n_model_inputs=2,
            model_input_sizes=[[2, 2], [2, 2]],
            dtype=torch.int32,
            n_sets_per_plan_test=n_sets_per_plan_test,
        )

        self.assertEqual(len(method_test_suites), len(method_names))

        # Compare to see if bundled testcases match expectations.
        for method_test_suite_idx in range(len(method_names)):
            self.assertEqual(
                method_test_suites[method_test_suite_idx].method_name,
                method_names[method_test_suite_idx],
            )
            for testset_idx in range(n_sets_per_plan_test):
                ri = rand_inputs[method_test_suite_idx][testset_idx]
                self.assertIOListEqual(
                    # pyre-ignore [6]: expected `List[Union[bool, float, int, Tensor]]` but got `Sequence[Union[bool, float, int, Tensor]]
                    ri,
                    method_test_suites[method_test_suite_idx]
                    .test_cases[testset_idx]
                    .inputs,
                )

                model_outputs = getattr(
                    eager_model, method_names[method_test_suite_idx]
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
                    method_test_suites[method_test_suite_idx]
                    .test_cases[testset_idx]
                    .expected_outputs,
                )
