# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import random
from typing import Any, Dict, List, Tuple, Union

import executorch.exir as exir
import torch
from executorch.bundled_program.config import BundledConfig
from executorch.exir import CaptureConfig
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


class MISOModel(torch.nn.Module):
    """An example model with Multi-Input Single-Output"""

    def __init__(self) -> None:
        super().__init__()
        self.a: torch.Tensor = 3 * torch.ones(2, 2, dtype=torch.int32)
        self.b: torch.Tensor = 2 * torch.ones(2, 2, dtype=torch.int32)

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, a: int = DEFAULT_INT_INPUT
    ) -> torch.Tensor:
        z = x.clone()
        torch.mul(self.a, x, out=z)
        y = x.clone()
        torch.add(z, self.b, alpha=a, out=y)
        torch.add(y, q, out=y)
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


def get_rand_attachement(n_kv_pairs: int = -1) -> Dict[str, Any]:
    """Helper function to generate random bundled attachments."""

    if n_kv_pairs == -1:
        n_kv_pairs = random.randint(0, 10)

    # This list cotains differnet possible values for generated random attachment.
    # They are in differnet types on purpose, which demonstrate bundle program
    # supports multiple types of attachment values. It is worth noting that it is
    # just a hacky method to generate random bundled attachments. It does not mean
    # bundled program can only support listed values.

    rand_attachment_vals = [
        b"BUNDLED_VALUE_IN_BYTES",  # random bytes array
        "BUNDLED_VALUE_IN_STR",  # random strinåg
        1.2345,  # random float
        True,  # random bool
        1,  # random int
    ]

    return {
        "BUNDLED_ATTACHMENT_KEY_{}".format(i): random.choice(rand_attachment_vals)
        for i in range(n_kv_pairs)
    }


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
    List[List[InputValues]],
    List[List[OutputValues]],
    List[Dict[str, Any]],
    Dict[str, Any],
    BundledConfig,
]:
    """Helper function to generate config filled with random inputs and expected outputs.

    The return type of rand inputs is a List[List[InputValues]]. The inner list of
    InputValues represents all test sets for single execution plan, while the outer list
    is for multiple execution plans.

    Same for rand_expected_outputs.

    """

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

    rand_metadatas = [get_rand_attachement() for _ in range(n_execution_plan_tests)]

    rand_attachment = get_rand_attachement()

    return (
        rand_inputs,
        rand_expected_outputs,
        rand_metadatas,
        rand_attachment,
        BundledConfig(
            rand_inputs, rand_expected_outputs, rand_metadatas, **rand_attachment
        ),
    )


def get_random_config_with_eager_model(
    eager_model: torch.nn.Module,
    n_model_inputs: int,
    model_input_sizes: List[List[int]],
    dtype: torch.dtype,
    n_sets_per_plan_test: int,
    n_execution_plan_tests: int,
) -> Tuple[List[List[InputValues]], BundledConfig]:
    """Generate config filled with random inputs for each inference method given eager model

    The details of return type is the same as get_random_config_with_rand_io_lists.

    NOTE: Right now we do not support multiple inference methods per eager model. To simulate
    generating exepected output for different inference functions, we infer the same method
    multiple times.

    TODO(T143752810): Update the hacky method after we support multiple inference methods.
    """
    inputs = get_rand_input_values(
        n_tensors=n_model_inputs,
        sizes=model_input_sizes,
        n_int=1,
        dtype=dtype,
        n_sets_per_plan_test=n_sets_per_plan_test,
        n_execution_plan_tests=n_execution_plan_tests,
    )

    expected_outputs = [
        [[eager_model(*x)] for x in inputs[i]] for i in range(n_execution_plan_tests)
    ]

    metadatas = [get_rand_attachement() for _ in range(n_execution_plan_tests)]

    attachment = get_rand_attachement()

    return inputs, BundledConfig(inputs, expected_outputs, metadatas, **attachment)


def get_common_program() -> Tuple[Program, BundledConfig]:
    """Helper function to generate a sample BundledProgram with its config."""
    eager_model = MISOModel()
    # Trace to FX Graph.
    capture_input = (
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        DEFAULT_INT_INPUT,
    )
    program = (
        exir.capture(eager_model, capture_input, CaptureConfig(pt2_mode=True))
        .to_edge()
        .to_executorch()
        .program
    )
    _, bundled_config = get_random_config_with_eager_model(
        eager_model=eager_model,
        n_model_inputs=2,
        model_input_sizes=[[2, 2], [2, 2]],
        dtype=torch.int32,
        n_sets_per_plan_test=10,
        n_execution_plan_tests=len(program.execution_plan),
    )
    return program, bundled_config
