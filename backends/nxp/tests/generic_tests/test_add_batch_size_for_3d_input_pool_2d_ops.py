# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.aten_passes.add_batch_size_for_3d_input_pool_2d_ops import (
    AddBatchSizeFor3DInputPool2DOps,
)
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.ops_aliases import (
    AdaptiveAvgPool2D,
    AvgPool2D,
    GetItem,
    MaxPool2DWithIndices,
    ViewCopy,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import (
    AdaptiveAvgPool2dModule,
    AvgPool2dModule,
    MaxPool2dModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


def apply_individual_pass_and_compare(model, export_input, pool_op):
    exir_program_aten = torch.export.export(
        model,
        export_input,
    ).module()

    # Check that pool op is present and has 3D input
    assert graph_contains_any_of_ops(exir_program_aten.graph, [pool_op])
    nodes = list(exir_program_aten.graph.nodes)
    pool_node = nodes[1]
    assert pool_node.target == pool_op
    assert len(pool_node.meta["val"].shape) == 3
    pool_count_prev = sum(
        [n.target == pool_op for n in list(exir_program_aten.graph.nodes)]
    )

    # Check that reshape is not present before the pass
    reshape_count_before = sum(
        [
            n.target == torch.ops.aten.reshape.default
            for n in list(exir_program_aten.graph.nodes)
        ]
    )
    assert reshape_count_before == 0

    outputs_before = [o.detach().numpy() for o in exir_program_aten(*export_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [AddBatchSizeFor3DInputPool2DOps()])(
        exir_program_aten
    )

    # Make sure pool op is still in the model and has 4D input with batch size == 1
    assert graph_contains_any_of_ops(exir_program_aten.graph, [pool_op])
    nodes = list(exir_program_aten.graph.nodes)
    pool_node = nodes[2]
    assert pool_node.target == pool_op
    assert len(pool_node.meta["val"].shape) == 4 and pool_node.meta["val"].shape[0] == 1

    # Make sure there is `reshape` in the model.
    assert graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.reshape.default],
    )

    reshape_count_after = sum(
        [
            n.target == torch.ops.aten.reshape.default
            for n in list(exir_program_aten.graph.nodes)
        ]
    )

    pool_count_after = sum(
        [n.target == pool_op for n in list(exir_program_aten.graph.nodes)]
    )

    # Make sure the number of pool operators is the same
    assert pool_count_prev == pool_count_after

    # Make sure we added 2 reshape operations per pool operation (add batch + remove batch)
    assert reshape_count_after == 2 * pool_count_after

    outputs_after = [o.detach().numpy() for o in exir_program_aten(*export_input)]

    # Make sure the model still produces the exact same output
    assert len(outputs_before) == len(outputs_after)

    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i], rtol=1e-5, atol=1e-5)


class TestAddBatchSizeFor3DPoolOps:
    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((16, 32, 32), (16, 16), id="3D, output_size=(16, 16)"),
            pytest.param((32, 64, 64), (8, 8), id="3D, output_size=(8, 8)"),
        ],
    )
    def test_add_batch_size_adaptive_avgpool2d(self, input_shape, output_size):
        model = AdaptiveAvgPool2dModule(output_size=output_size)
        example_input = torch.rand(input_shape, dtype=torch.float32)
        apply_individual_pass_and_compare(
            model, (example_input,), torch.ops.aten.adaptive_avg_pool2d.default
        )

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding",
        [
            pytest.param(
                (16, 32, 32), 2, 2, 0, id="3D, kernel=2, stride=2, no padding"
            ),
            pytest.param((32, 64, 64), 3, 3, 1, id="3D, kernel=3, stride=3, padding=1"),
            pytest.param(
                (8, 16, 16), (2, 2), (2, 2), 0, id="3D, kernel=(2,2), stride=(2,2)"
            ),
        ],
    )
    def test_add_batch_size_avgpool2d(self, input_shape, kernel_size, stride, padding):
        model = AvgPool2dModule(kernel_size=kernel_size, stride=stride, padding=padding)
        example_input = torch.rand(input_shape, dtype=torch.float32)
        apply_individual_pass_and_compare(
            model, (example_input,), torch.ops.aten.avg_pool2d.default
        )

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding",
        [
            pytest.param(
                (16, 32, 32), 2, 2, 0, id="3D, kernel=2, stride=2, no padding"
            ),
            pytest.param((32, 64, 64), 3, 3, 1, id="3D, kernel=3, stride=3, padding=1"),
            pytest.param(
                (8, 16, 16), (2, 2), (2, 2), 0, id="3D, kernel=(2,2), stride=(2,2)"
            ),
        ],
    )
    def test_add_batch_size_maxpool2d(self, input_shape, kernel_size, stride, padding):
        model = MaxPool2dModule(kernel_size=kernel_size, stride=stride, padding=padding)
        example_input = torch.rand(input_shape, dtype=torch.float32)
        apply_individual_pass_and_compare(
            model, (example_input,), torch.ops.aten.max_pool2d.default
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((1, 16, 32, 32), id="4D input - should not transform"),
            pytest.param(
                (2, 16, 32, 32), id="4D input with batch=2 - should not transform"
            ),
        ],
    )
    @pytest.mark.parametrize(
        "model, pool_op",
        [
            pytest.param(
                AdaptiveAvgPool2dModule(output_size=(16, 16)),
                torch.ops.aten.adaptive_avg_pool2d.default,
                id="AdaptiveAvgPoolModel",
            ),
            pytest.param(
                AvgPool2dModule(kernel_size=2, stride=2, padding=0),
                torch.ops.aten.avg_pool2d.default,
                id="AvgPoolModel",
            ),
            pytest.param(
                MaxPool2dModule(kernel_size=2, stride=2, padding=0),
                torch.ops.aten.max_pool2d.default,
                id="MaxPoolModel",
            ),
        ],
    )
    def test_no_transform_for_4d_input(self, input_shape, model, pool_op):
        example_input = torch.rand(input_shape, dtype=torch.float32)

        exir_program_aten = torch.export.export(
            model,
            (example_input,),
        ).module()

        # Check that pool op is present
        assert graph_contains_any_of_ops(
            exir_program_aten.graph,
            [pool_op],
        )

        # Check that reshape is not present before the pass
        reshape_count_before = sum(
            [
                n.target == torch.ops.aten.reshape.default
                for n in list(exir_program_aten.graph.nodes)
            ]
        )
        assert reshape_count_before == 0

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec, [AddBatchSizeFor3DInputPool2DOps()]
        )(exir_program_aten)

        # Check that reshape count hasn't changed (no transformation for 4D input)
        reshape_count_after = sum(
            [
                n.target == torch.ops.aten.reshape.default
                for n in list(exir_program_aten.graph.nodes)
            ]
        )

        assert reshape_count_before == reshape_count_after

    @pytest.mark.parametrize(
        "input_shape",
        [
            (16, 32, 32),
            (32, 64, 64),
        ],
        ids=lambda shape: f"3D_{shape[0]}x{shape[1]}x{shape[2]}",
    )
    @pytest.mark.parametrize(
        "pool_type",
        ["adaptive_avg", "avg", "max"],
        ids=lambda pool_type: f"{pool_type}pool2d",
    )
    def test__3d_pool__full_pipeline(
        self, mocker, request, input_shape: tuple[int, ...], pool_type: str
    ):
        expected_delegated_ops = {}
        match pool_type:
            case "adaptive_avg":
                model = AdaptiveAvgPool2dModule(output_size=(16, 16))
                expected_delegated_ops = {ViewCopy: 2, AdaptiveAvgPool2D: 1}
            case "avg":
                model = AvgPool2dModule(kernel_size=2, stride=2, padding=0)
                expected_delegated_ops = {ViewCopy: 2, AvgPool2D: 1}
            case _:
                model = MaxPool2dModule(kernel_size=2, stride=2, padding=0)
                expected_delegated_ops = {
                    ViewCopy: 2,
                    MaxPool2DWithIndices: 1,
                    GetItem: 1,
                }

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        dataset_creator = RandomDatasetCreator(low=-1, high=1)

        remove_quant_io_ops = True  # Use quantized dataset.
        output_comparator = AllCloseOutputComparator(atol=1)  # Allow single bit error.

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            output_comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )
