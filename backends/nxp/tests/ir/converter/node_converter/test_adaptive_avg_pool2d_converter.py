# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch
from executorch.backends.nxp.ops_aliases import (
    AdaptiveAvgPool2D,
    ExecutorchDelegateCall,
    ViewCopy,
)

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import (
    AdaptiveAvgPool1dModule,
    AdaptiveAvgPool2dModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestAdaptiveAvgPool2D:
    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 3, 16, 16), (8, 8), id="H == W."),
            pytest.param((1, 3, 16, 8), (8, 2), id="H != W."),
            pytest.param(
                (2, 3, 4, 6),
                (2, 3),
                id="H != W, non multiples of num_macs, batch != 1.",
            ),
            pytest.param(
                (2, 3, 10, 15),
                (5, 5),
                id="H != W, non multiples of num_macs, batch != 1, fixed fail.",
            ),
        ],
    )
    def test__basic_nsys_inference(
        self, mocker, request, use_qat, input_shape, output_size
    ):
        model = AdaptiveAvgPool2dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1},
            expected_non_delegated_ops={},
        )

        remove_quant_io_ops = True  # Use quantized dataset.
        output_comparator = AllCloseOutputComparator(atol=1)  # Allow single bit error.

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    def test__kernel_size_and_stride_limit(self, mocker, request):
        input_shape = (1, 3, 4, 4096)  # input_size = (1, 4096)
        output_size = (
            2,
            1,
        )  # If we reduced both dims to 1, ExecuTorch would replace the op with mean.
        # stride = input_size // output_size = 4096 / 1 = 4096
        # kernel_size = input_size - (output_size - 1) * stride = 4096 - 0 * 4096 = 4096

        model = AdaptiveAvgPool2dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1},
            expected_non_delegated_ops={},
        )

        remove_quant_io_ops = True  # Use quantized dataset.
        output_comparator = AllCloseOutputComparator(atol=1)  # Allow single bit error.

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    def test__kernel_size_and_stride_limit_exceeded(self):
        input_shape = (1, 3, 4, 4097)  # input_size = (1, 4097)
        output_size = (
            2,
            1,
        )  # If we reduced both dims to 1, ExecuTorch would replace the op with mean.
        # stride = input_size // output_size = 4097 / 1 = 4097
        # kernel_size = input_size - (output_size - 1) * stride = 4097 - 0 * 4097 = 4097

        model = AdaptiveAvgPool2dModule(output_size)
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `adaptive_avg_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AdaptiveAvgPool2D])


class TestAdaptiveAvgPool1DTo2D:

    # Just a basic test to verify that the operator gets extended to the 2D variant correctly.
    def test__basic_nsys_inference(self, mocker, request, use_qat):
        input_shape = (2, 4, 6)  # The old flow limited the batch size to 1.
        output_size = (3,)
        model = AdaptiveAvgPool1dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1, ViewCopy: 2},
            expected_non_delegated_ops={},
        )

        remove_quant_io_ops = True  # Use quantized dataset.
        output_comparator = AllCloseOutputComparator(atol=1)  # Allow single bit error.

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )
