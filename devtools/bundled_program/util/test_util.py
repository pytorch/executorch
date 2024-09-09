# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import random
import string
from typing import List, Tuple

import torch
from executorch.devtools.bundled_program.config import (
    MethodInputType,
    MethodOutputType,
    MethodTestCase,
    MethodTestSuite,
)

from executorch.exir import ExecutorchProgramManager, to_edge
from torch.export import export
from torch.export.unflatten import _assign_attr, _AttrKind

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


class SampleModel(torch.nn.Module):
    """An example model with multi-methods. Each method has multiple input and single output"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("a", 3 * torch.ones(2, 2, dtype=torch.int32))
        self.register_buffer("b", 2 * torch.ones(2, 2, dtype=torch.int32))
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
    n_method_test_suites: int,
) -> List[List[MethodInputType]]:
    # pyre-ignore[7]: expected `List[List[List[Union[bool, float, int, Tensor]]]]` but got `List[List[List[Union[int, Tensor]]]]`
    return [
        [
            [(torch.rand(*sizes[i]) - 0.5).to(dtype) for i in range(n_tensors)]
            + [DEFAULT_INT_INPUT for _ in range(n_int)]
            for _ in range(n_sets_per_plan_test)
        ]
        for _ in range(n_method_test_suites)
    ]


def get_rand_output_values(
    n_tensors: int,
    sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_method_test_suites: int,
) -> List[List[MethodOutputType]]:
    # pyre-ignore [7]: Expected `List[List[Sequence[Tensor]]]` but got `List[List[List[Tensor]]]`.
    return [
        [
            [(torch.rand(*sizes[i]) - 0.5).to(dtype) for i in range(n_tensors)]
            for _ in range(n_sets_per_plan_test)
        ]
        for _ in range(n_method_test_suites)
    ]


def get_rand_method_names(n_method_test_suites: int) -> List[str]:
    unique_strings = set()
    while len(unique_strings) < n_method_test_suites:
        rand_str = "".join(random.choices(string.ascii_letters, k=5))
        if rand_str not in unique_strings:
            unique_strings.add(rand_str)
    return list(unique_strings)


def get_random_test_suites(
    n_model_inputs: int,
    model_input_sizes: List[List[int]],
    n_model_outputs: int,
    model_output_sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_method_test_suites: int,
) -> Tuple[
    List[str],
    List[List[MethodInputType]],
    List[List[MethodOutputType]],
    List[MethodTestSuite],
]:
    """Helper function to generate config filled with random inputs and expected outputs.

    The return type of rand inputs is a List[List[InputValues]]. The inner list of
    InputValues represents all test sets for single execution plan, while the outer list
    is for multiple execution plans.

    Same for rand_expected_outputs.

    """

    rand_method_names = get_rand_method_names(n_method_test_suites)

    rand_inputs_per_program = get_rand_input_values(
        n_tensors=n_model_inputs,
        sizes=model_input_sizes,
        n_int=1,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_method_test_suites=n_method_test_suites,
    )

    rand_expected_output_per_program = get_rand_output_values(
        n_tensors=n_model_outputs,
        sizes=model_output_sizes,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_method_test_suites=n_method_test_suites,
    )

    rand_method_test_suites: List[MethodTestSuite] = []

    for (
        rand_method_name,
        rand_inputs_per_method,
        rand_expected_output_per_method,
    ) in zip(
        rand_method_names, rand_inputs_per_program, rand_expected_output_per_program
    ):
        rand_method_test_cases: List[MethodTestCase] = []
        for rand_inputs, rand_expected_outputs in zip(
            rand_inputs_per_method, rand_expected_output_per_method
        ):
            rand_method_test_cases.append(
                MethodTestCase(
                    inputs=rand_inputs, expected_outputs=rand_expected_outputs
                )
            )

        rand_method_test_suites.append(
            MethodTestSuite(
                method_name=rand_method_name, test_cases=rand_method_test_cases
            )
        )

    return (
        rand_method_names,
        rand_inputs_per_program,
        rand_expected_output_per_program,
        rand_method_test_suites,
    )


def get_random_test_suites_with_eager_model(
    eager_model: torch.nn.Module,
    method_names: List[str],
    n_model_inputs: int,
    model_input_sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
) -> Tuple[List[List[MethodInputType]], List[MethodTestSuite]]:
    """Generate config filled with random inputs for each inference method given eager model

    The details of return type is the same as get_random_test_suites_with_rand_io_lists.
    """
    inputs_per_program = get_rand_input_values(
        n_tensors=n_model_inputs,
        sizes=model_input_sizes,
        n_int=1,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_method_test_suites=len(method_names),
    )

    method_test_suites: List[MethodTestSuite] = []

    for method_name, inputs_per_method in zip(method_names, inputs_per_program):
        method_test_cases: List[MethodTestCase] = []
        for inputs in inputs_per_method:
            method_test_cases.append(
                MethodTestCase(
                    inputs=inputs,
                    expected_outputs=getattr(eager_model, method_name)(*inputs),
                )
            )

        method_test_suites.append(
            MethodTestSuite(method_name=method_name, test_cases=method_test_cases)
        )

    return inputs_per_program, method_test_suites


class StatefulWrapperModule(torch.nn.Module):
    """A version of wrapper module that preserves parameters/buffers.

    Use this if you are planning to wrap a non-forward method on an existing
    module.
    """

    def __init__(self, base_mod, method) -> None:  # pyre-ignore
        super().__init__()
        state_dict = base_mod.state_dict()
        for name, value in base_mod.named_parameters():
            _assign_attr(value, self, name, _AttrKind.PARAMETER)
        for name, value in base_mod.named_buffers():
            _assign_attr(
                value, self, name, _AttrKind.BUFFER, persistent=name in state_dict
            )
        self.fn = method  # pyre-ignore

    def forward(self, *args, **kwargs):  # pyre-ignore
        return self.fn(*args, **kwargs)


def get_common_executorch_program() -> (
    Tuple[ExecutorchProgramManager, List[MethodTestSuite]]
):
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

    # Trace to FX Graph and emit the program
    method_graphs = {
        m_name: export(
            StatefulWrapperModule(eager_model, getattr(eager_model, m_name)),
            capture_inputs[m_name],
        )
        for m_name in eager_model.method_names
    }

    executorch_program = to_edge(method_graphs).to_executorch()

    _, method_test_suites = get_random_test_suites_with_eager_model(
        eager_model=eager_model,
        method_names=eager_model.method_names,
        n_model_inputs=2,
        model_input_sizes=[[2, 2], [2, 2]],
        dtype=torch.int32,
        n_sets_per_plan_test=10,
    )
    return executorch_program, method_test_suites
