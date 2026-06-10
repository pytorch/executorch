# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dWithActivation, HardTanhModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Convolution
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
HardTanh = exir_ops.edge.aten.hardtanh.default


class TestHardTanhNewNeutronFlow:
    @pytest.mark.parametrize("input_shape", [(1, 3, 128, 128)])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_relu6_quant(
        self, mocker, input_shape: tuple[int], inplace: bool, use_qat: bool
    ):
        # The torch.nn.Relu6 inherits from torch.nn.Hardtanh, and hence represented as HardTanh in ATen.
        # Testing the hardtanh originated from torch.nn.Relu6 op.
        model = Conv2dWithActivation(
            activation=torch.nn.ReLU6(inplace=inplace), in_channels=input_shape[1]
        )

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={HardTanh: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model=model,
            input_spec=input_shape,
            dlg_model_verifier=graph_verifier,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize("input_shape", [(1, 3, 16, 16), (1, 3, 32, 32)])
    @pytest.mark.parametrize(
        "activation_range",
        [
            (0.0, 6.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (0.0, float("inf")),
            (0, 6),
            (-1, 1),
            (0, 1),
            (0, float("inf")),
        ],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_custom_hardtanh_quant(
        self,
        mocker,
        input_shape: tuple[int],
        activation_range: tuple[float, float],
        inplace: bool,
        use_qat: bool,
    ):
        min_val, max_val = activation_range
        model = Conv2dWithActivation(
            activation=torch.nn.Hardtanh(
                min_val=min_val, max_val=max_val, inplace=inplace
            ),
            in_channels=input_shape[1],
        )

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={HardTanh: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model=model,
            input_spec=input_shape,
            dlg_model_verifier=graph_verifier,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape, activation_range",
        [
            pytest.param(
                (3, 7, 15, 7),
                (0, float("inf")),
                id="activation range: Relu, num_channels not divisible by NUM_MACS, alone in partition",
            ),
            pytest.param(
                (3, 7, 15, 7),
                (0, 6),
                id="activation range: Relu6, num_channels not divisible by NUM_MACS, alone in partition",
            ),
        ],
    )
    def test_hardtanh__old_flow_unsupported(
        self,
        mocker,
        input_shape: tuple[int],
        activation_range: tuple[float, float],
        use_qat: bool,
    ):
        min_val, max_val = activation_range
        model = HardTanhModule(min_val, max_val)

        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={HardTanh: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model=model,
            input_spec=input_shape,
            dlg_model_verifier=graph_verifier,
            dataset_creator=RandomDatasetCreator(low=-1, high=1),
            use_qat=use_qat,
        )
