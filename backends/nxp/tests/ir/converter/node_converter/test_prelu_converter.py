# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import (
    ConvPReLUModule,
    LinearPReLUModule,
    TwoPartitionPReLUModel,
)

from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddMM,
    Convolution,
    ExecutorchDelegateCall,
    GtScalar,
    MulTensor,
    PermuteCopy,
    Prelu,
    ViewCopy,
    WhereSelf,
)
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestPreluConverter:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        request,
        expected_delegated_ops=None,
        use_qat=False,
    ):
        rank = len(input_shape)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops
            or {
                Prelu: 1,
                AddMM: 1,
                PermuteCopy: 1,
                ViewCopy: 0 if rank == 2 else 2,
            },
            expected_non_delegated_ops={},
        )
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            output_comparator=comparator,
            remove_quant_io_ops=True,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param(
                (36, 487),
                id="2D incorrect results.",
                marks=pytest.mark.xfail(
                    reason="AIR-14737: incorrect results", strict=True
                ),
            ),
            pytest.param(
                (87, 842),
                id="2D incorrect results alt.",
                marks=pytest.mark.xfail(
                    reason="AIR-14737: incorrect results", strict=True
                ),
            ),
            pytest.param((7, 83), id="2D."),
            pytest.param(
                (1, 43, 183),
                id="3D incorrect results alt.",
                marks=pytest.mark.xfail(
                    reason="AIR-14737: incorrect results", strict=True
                ),
            ),
            pytest.param((1, 43, 93), id="3D."),
            pytest.param((1, 4, 7, 8), id="4D."),
            pytest.param((1, 4, 3, 4, 14), id="5D."),
        ],
    )
    def test__basic_nsys_inference(self, mocker, request, input_shape):
        channels = input_shape[-1]
        model = LinearPReLUModule(in_features=channels, out_features=channels)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

        self.assert_delegated(model, input_shape, mocker, request)

        # Capture generated entities
        exported_program: ExportedProgram = converter_spy.call_args.args[1]

        # Check `prelu` was not decomposed into simpler edge operators
        assert not graph_contains_any_of_ops(
            exported_program.graph,
            [
                GtScalar,
                MulTensor,
                WhereSelf,
            ],
        )

    def test__basic_nsys_inference_qat(self, mocker, request):
        input_shape = (2, 4, 6, 7)
        channels = input_shape[-1]
        model = LinearPReLUModule(in_features=channels, out_features=channels)

        self.assert_delegated(model, input_shape, mocker, request, use_qat=True)

    def test__num_parameters_param(self, mocker, request):
        input_shape = (1, 43, 93)
        channels = input_shape[-1]
        num_parameters = input_shape[1]
        model = LinearPReLUModule(
            in_features=channels, out_features=channels, num_parameters=num_parameters
        )

        self.assert_delegated(model, input_shape, mocker, request)

    def test_prelu_2_partitions(self):
        input_shape = (1, 8, 24, 32)
        # Run conversion
        edge_program = to_quantized_edge_program(
            TwoPartitionPReLUModel(), [input_shape, input_shape]
        ).exported_program()

        # Check `prelu` was delegated
        assert not graph_contains_any_of_ops(
            edge_program.graph,
            [Prelu],
        )

        # Check there are two partitions
        edge_nodes = list(edge_program.graph.nodes)
        assert sum(n.target == ExecutorchDelegateCall for n in edge_nodes) == 2

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((1, 8, 42, 24), id="4D."),
            pytest.param(
                (1, 8, 42, 21),
                id="4D incorrect results.",
                marks=pytest.mark.xfail(
                    reason="AIR-14737: incorrect results", strict=True
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "num_parameters_channels",
        [True, False],
        ids=lambda ch: "Num parameters channels" if ch else "Num parameters 1",
    )
    def test__channels_first(
        self, mocker, request, input_shape, num_parameters_channels
    ):
        channels = input_shape[1]
        num_parameters = channels if num_parameters_channels else 1
        model = ConvPReLUModule(in_channels=channels, num_parameters=num_parameters)
        expected_delegated_ops = {
            Prelu: 1,
            Convolution: 1,
        }

        self.assert_delegated(
            model, input_shape, mocker, request, expected_delegated_ops
        )
