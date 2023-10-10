# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import random
import string
from typing import List, Tuple, Union

import executorch.exir as exir
import torch
from executorch.bundled_program.config import BundledConfig
from executorch.exir.schema import Program

# @manual=fbsource//third-party/pypi/typing-extensions:typing-extensions
from typing_extensions import TypeAlias

# A hacky integer to deal with a mismatch between execution plan and complier.
#
# Execution plan supports multiple types of inputs, like Tensor, Int, etc,
# rather than only Tensor. However, compiler only supports Tensor as input type.
# All other inputs will remain the same as default value in the model, which
# means during model execution, each function will use the preset default value
# for non-tensor inputs, rather than the one we manually set. However, eager
# model supports multiple types of inputs.
#
# In order to show that bundled program can support multiple input types while
# executorch model can generate the same output as eager model, we hackily set
# all Int inputs in Bundled Program and default int inputs in model as a same
# value, called DEFAULT_INT_INPUT.
#
# TODO(gasoonjia): track the situation. Stop supporting multiple input types in
# bundled program if execution plan stops supporting it, or remove this hacky
# method if compiler can support multiple input types
DEFAULT_INT_INPUT = 2


# Alias type for all datas model needs for single execution.
InputValues: TypeAlias = List[Union[torch.Tensor, int]]

# Alias type for all datas model generates per execution.
OutputValues: TypeAlias = List[torch.Tensor]


class SampleModel(torch.nn.Module):
    """An example model with multi-methods. Each method has multiple input and single output"""

    def __init__(self) -> None:
        super().__init__()
        self.a: torch.Tensor = 3 * torch.ones(2, 2, dtype=torch.int32)
        self.b: torch.Tensor = 2 * torch.ones(2, 2, dtype=torch.int32)
        self.method_names = ["encode", "decode"]

    def encode(
        self, x: torch.Tensor, q: torch.Tensor, a: int = DEFAULT_INT_INPUT
    ) -> torch.Tensor:
        z = x.clone()
        torch.mul(self.a, x, out=z)
        y = x.clone()
        torch.add(z, self.b, alpha=a, out=y)
        torch.add(y, q, out=y)
        return y

    def decode(
        self, x: torch.Tensor, q: torch.Tensor, a: int = DEFAULT_INT_INPUT
    ) -> torch.Tensor:
        y = x * q
        torch.add(y, self.b, alpha=a, out=y)
        return y


def get_rand_input_values(
    n_tensors: int,
    sizes: List[List[int]],
    n_int: int,
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_execution_plan_tests: int,
) -> List[List[InputValues]]:
    return [
        [
            [(torch.rand(*sizes[i]) - 0.5).to(dtype) for i in range(n_tensors)]
            + [DEFAULT_INT_INPUT for _ in range(n_int)]
            for _ in range(n_sets_per_plan_test)
        ]
        for _ in range(n_execution_plan_tests)
    ]


def get_rand_output_values(
    n_tensors: int,
    sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_execution_plan_tests: int,
) -> List[List[OutputValues]]:
    return [
        [
            [(torch.rand(*sizes[i]) - 0.5).to(dtype) for i in range(n_tensors)]
            for _ in range(n_sets_per_plan_test)
        ]
        for _ in range(n_execution_plan_tests)
    ]


def get_rand_method_names(n_execution_plan_tests: int) -> List[str]:
    unique_strings = set()
    while len(unique_strings) < n_execution_plan_tests:
        rand_str = "".join(random.choices(string.ascii_letters, k=5))
        if rand_str not in unique_strings:
            unique_strings.add(rand_str)
    return list(unique_strings)


# TODO(T143955558): make n_int and metadatas as its input;
def get_random_config(
    n_model_inputs: int,
    model_input_sizes: List[List[int]],
    n_model_outputs: int,
    model_output_sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_execution_plan_tests: int,
) -> Tuple[
    List[str],
    List[List[InputValues]],
    List[List[OutputValues]],
    BundledConfig,
]:
    """Helper function to generate config filled with random inputs and expected outputs.

    The return type of rand inputs is a List[List[InputValues]]. The inner list of
    InputValues represents all test sets for single execution plan, while the outer list
    is for multiple execution plans.

    Same for rand_expected_outputs.

    """

    rand_method_names = get_rand_method_names(n_execution_plan_tests)

    rand_inputs = get_rand_input_values(
        n_tensors=n_model_inputs,
        sizes=model_input_sizes,
        n_int=1,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_execution_plan_tests=n_execution_plan_tests,
    )

    rand_expected_outputs = get_rand_output_values(
        n_tensors=n_model_outputs,
        sizes=model_output_sizes,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_execution_plan_tests=n_execution_plan_tests,
    )

    return (
        rand_method_names,
        rand_inputs,
        rand_expected_outputs,
        # pyre-ignore[6]: Expected Union[Tensor, int, float, bool] for each element in 2nd positional argument, but got Union[Tensor, int]
        BundledConfig(rand_method_names, rand_inputs, rand_expected_outputs),
    )


def get_random_config_with_eager_model(
    eager_model: torch.nn.Module,
    method_names: List[str],
    n_model_inputs: int,
    model_input_sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
) -> Tuple[List[List[InputValues]], BundledConfig]:
    """Generate config filled with random inputs for each inference method given eager model

    The details of return type is the same as get_random_config_with_rand_io_lists.
    """
    inputs = get_rand_input_values(
        n_tensors=n_model_inputs,
        sizes=model_input_sizes,
        n_int=1,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_execution_plan_tests=len(method_names),
    )

    expected_outputs = [
        [[getattr(eager_model, m_name)(*x)] for x in inputs[i]]
        for i, m_name in enumerate(method_names)
    ]

    # pyre-ignore[6]: Expected Union[Tensor, int, float, bool] for each element in 2nd positional argument, but got Union[Tensor, int]
    return inputs, BundledConfig(method_names, inputs, expected_outputs)


def get_common_program() -> Tuple[Program, BundledConfig]:
    """Helper function to generate a sample BundledProgram with its config."""
    eager_model = SampleModel()
    # Trace to FX Graph.
    capture_inputs = {
        m_name: (
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
            DEFAULT_INT_INPUT,
        )
        for m_name in eager_model.method_names
    }

    program = (
        exir.capture_multiple(eager_model, capture_inputs)
        .to_edge()
        .to_executorch()
        .program
    )
    _, bundled_config = get_random_config_with_eager_model(
        eager_model=eager_model,
        method_names=eager_model.method_names,
        n_model_inputs=2,
        model_input_sizes=[[2, 2], [2, 2]],
        dtype=torch.int32,
        n_sets_per_plan_test=10,
    )
    return program, bundled_config
