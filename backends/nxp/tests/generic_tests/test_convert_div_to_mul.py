# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    ConvertDivToMulPass,
    NeutronAtenPassManager,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import (
    NonstaticDivLinearModel,
    StaticDivLinearModel,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import MulTensor


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


def apply_individual_pass_and_compare(model, export_input, compare_input=None):
    exir_program_aten = torch.export.export(
        model,
        export_input,
    ).module()

    # Check `aten.div` is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.div.Tensor]
    )
    div_count_prev = sum(
        [
            n.target in [torch.ops.aten.div.Tensor]
            for n in list(exir_program_aten.graph.nodes)
        ]
    )

    if compare_input is None:
        outputs_before = [o.detach().numpy() for o in exir_program_aten(*export_input)]
    else:
        outputs_before = [o.detach().numpy() for o in exir_program_aten(compare_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [ConvertDivToMulPass()])(
        exir_program_aten
    )

    # Make sure no `aten.div` is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.div.Tensor],
    )

    # Make sure there is `aten.mul` in the model.
    assert graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.mul.Tensor],
    )

    mul_count_prev = sum(
        [
            n.target == torch.ops.aten.mul.Tensor
            for n in list(exir_program_aten.graph.nodes)
        ]
    )

    # Make sure the number of converted `mul` operators is the same as original `div` operators
    assert div_count_prev == mul_count_prev

    if compare_input is None:
        outputs_after = [o.detach().numpy() for o in exir_program_aten(*export_input)]
    else:
        outputs_after = [o.detach().numpy() for o in exir_program_aten(compare_input)]

    # Make sure the model still produces the exact same output
    assert len(outputs_before) == len(outputs_after)

    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


@pytest.mark.parametrize(
    "input_shape, is_scalar",
    [
        pytest.param((8,), True, id="1D, scalar."),
        pytest.param((8, 8, 16), True, id="3D, scalar."),
        pytest.param((8,), False, id="1D, tensor."),
        pytest.param((16, 8, 16), False, id="3D, tensor."),
    ],
)
def test_convert_div_to_mul_static(mocker, input_shape, is_scalar):
    channels = input_shape[-1]
    if is_scalar:
        divisor = np.random.uniform(0, 15)
        model = StaticDivLinearModel(
            in_channels=channels, out_channels=channels, divisor=divisor
        )
    else:
        divisor = torch.rand(input_shape)
        model = StaticDivLinearModel(
            in_channels=channels, out_channels=channels, divisor=divisor
        )

    example_input = torch.rand(input_shape, dtype=torch.float32)
    apply_individual_pass_and_compare(model, (example_input,), example_input)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((8,), id="1D, scalar."),
        pytest.param((8, 8, 16), id="3D, scalar."),
    ],
)
def test_convert_div_to_mul_nonstatic_scalar(mocker, input_shape):
    channels = input_shape[-1]
    model = NonstaticDivLinearModel(in_channels=channels, out_channels=channels)

    divisor = np.random.uniform(0, 15)
    example_input = (
        torch.rand(input_shape, dtype=torch.float32),
        divisor,
    )
    apply_individual_pass_and_compare(model, example_input)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((8, 8, 16), id="3D."),
    ],
)
def test_convert_div_to_mul_non_static_tensor(mocker, input_shape):
    channels = input_shape[-1]
    model = NonstaticDivLinearModel(in_channels=channels, out_channels=channels)

    divisor = torch.rand(input_shape, dtype=torch.float32)
    example_input = (
        torch.rand(input_shape, dtype=torch.float32),
        divisor,
    )
    exir_program_aten = torch.export.export(
        model,
        example_input,
    ).module()

    # Check `aten.div` is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.div.Tensor]
    )

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [ConvertDivToMulPass()])(
        exir_program_aten
    )

    # Make sure `aten.div` is still in the model.
    assert graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.div.Tensor],
    )

    # Make sure there is no `aten.mul` in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.mul.Tensor],
    )


class StaticDivModel(torch.nn.Module):
    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x):
        return x / self.divisor


class TestConvertDivToMul:

    @pytest.mark.parametrize(
        "input_shape",
        [
            (23,),
            (3, 7),
            (2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 2, 1),
        ],
        ids=lambda shape: f"{len(shape)}D",
    )
    @pytest.mark.parametrize(
        "is_scalar",
        [False, True],
        ids=lambda is_scalar: "scalar" if is_scalar else "tensor",
    )
    def test__static__full_pipeline(
        self, mocker, request, input_shape: tuple[int, ...], is_scalar: bool
    ):
        if is_scalar:
            divisor = np.random.uniform(0.01, 15)
            model = StaticDivModel(divisor)
        else:
            divisor = torch.rand(input_shape) + 0.01
            model = StaticDivModel(divisor)

        graph_verifier = DetailedGraphVerifier(
            mocker,
            # By the time `DetailedGraphVerifier` checks for operators, the `div` has already been replaced by `mul`.
            expected_delegated_ops={MulTensor: 1},
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
        )
