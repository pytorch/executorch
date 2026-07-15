# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    EdgeProgramToIRConverter,
    ExportedProgram,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dModule
from executorch.backends.nxp.tests.nsys_testing import (
    AllCloseOutputComparator,
    lower_run_compare,
)
from executorch.backends.nxp.tests.ops_aliases import (
    Convolution,
    ExecutorchDelegateCall,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestTransposedConvFromLegacyFlow:
    @pytest.mark.parametrize(
        "model, input_shape",
        [
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 4), stride=(1, 2)),
                (1, 8, 1, 16),
                id="In ch 8, out ch 16, kernel (1, 4), stride (1, 2)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(64, 64, (1, 2), stride=(1, 2)),
                (1, 64, 3, 12),
                id="In ch 64, out ch 64, kernel (1, 2), stride (1, 2)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(16, 40, (1, 4), stride=(1, 2), padding=(0, 1)),
                (1, 16, 1, 27),
                id="In ch 16, out ch 40, kernel (1, 4), stride (1, 2), padding (0, 1)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 4), stride=(1, 2), padding=(0, 1)),
                (1, 8, 1, 16),
                id="In ch 8, out ch 16, kernel (1, 4), stride (1, 2), padding (0, 1)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(
                    8, 16, (1, 4), stride=(1, 2), output_padding=(0, 1)
                ),
                (1, 8, 1, 16),
                id="In ch 8, out ch 16, kernel (1, 8), stride (1, 2), output_padding (0, 1)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(16, 16, (1, 4), stride=(1, 2)),
                (1, 16, 1, 16),
                id="In ch 16, out ch 16, kernel (1, 4), stride (1, 2)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 4), stride=(1, 2), bias=False),
                (1, 8, 1, 16),
                id="In ch 8, out ch 16, kernel (1, 4), stride (1, 2), no bias",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(
                    8, 16, (1, 4), stride=(1, 2), padding=(0, 1), bias=False
                ),
                (1, 8, 1, 16),
                id="In ch 8, out ch 16, kernel (1, 4), stride (1, 2),"
                "padding (0, 1), no bias",
            ),
        ],
    )
    def test_conv_transpose2d_conversion__quantized(
        self, mocker, model: torch.nn.Module, input_shape, use_qat
    ):
        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

        edge_program = to_quantized_edge_program(
            model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
        ).exported_program()

        # Make sure the `TransposeConv` was delegated.
        assert not graph_contains_any_of_ops(
            graph=edge_program.graph, ops=[Convolution]
        )
        assert graph_contains_any_of_ops(
            graph=edge_program.graph, ops=[ExecutorchDelegateCall]
        )

        # Capture generated model
        tflite_flatbuffers_model, *_ = converter_spy.spy_return

        # Capture converted program
        exported_program: ExportedProgram = converter_spy.call_args.args[1]

        input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
            np.int8
        )

        convert_run_compare(
            exported_program,
            tflite_input_preprocess=ToChannelLastPreprocess(),
            tfl_model=tflite_flatbuffers_model,
            tflite_output_preprocess=ToChannelFirstPreprocess(),
            input_data=input_data,
            atol=1.0,
        )

    @pytest.mark.parametrize(
        "model, input_shape",
        [
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 4), stride=(1, 2), dilation=(1, 2)),
                (1, 8, 1, 16),
                id="Dilation != (1, 1)",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(6, 16, (1, 4), stride=(1, 2)),
                (1, 6, 1, 16),
                id="In channels % num_macs != 0",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 4), stride=(1, 2)),
                (1, 8, 4, 16),
                id="Out height != 1, stride width != kernel width",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (2, 4), stride=(1, 2), padding=(0, 1)),
                (1, 8, 1, 16),
                id="Out height != 1, stride width != kernel width",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(8, 16, (1, 5), stride=(1, 4)),
                (1, 8, 1, 16),
                id="Stride width != kernel width / 2, stride width != kernel width",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(16, 12, (1, 4), stride=(3, 3)),
                (1, 16, 1, 16),
                id="Out channels % num_macs != 0",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(64, 64, (1, 4), stride=(1, 2)),
                (1, 64, 3, 12),
                id="Out height != 1, stride width != kernel width",
            ),
            pytest.param(
                torch.nn.ConvTranspose2d(16, 40, (1, 4), stride=(1, 4), padding=(0, 1)),
                (1, 16, 4, 27),
                id="Padding width != 1 and input height != 1",
            ),
        ],
    )
    def test_conv_transpose2d_non_delegated_conversion__quantized(
        self, model: torch.nn.Module, input_shape, use_qat
    ):
        edge_program = to_quantized_edge_program(
            model, input_shape, use_qat=use_qat
        ).exported_program()

        nodes = list(edge_program.graph.nodes)
        assert len(nodes) == 15
        assert nodes[11].target == Convolution  # TransposeConv not delegated.


class TestConv:
    @staticmethod
    def _conv_id(ins, oc, ks=3, s=2, d=1, p=0, b=True, g=1):
        return (
            f"ic={ins}, "
            f"oc={oc}, "
            f"ks={ks}, "
            f"s={s}, "
            f"d={d}, "
            f"p={p}, "
            f"b={b}, "
            f"g={g}"
        )

    @staticmethod
    def assert_delegated_and_correct(model, input_shape, mocker, request, use_qat):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Convolution: 1},
            expected_non_delegated_ops={},
        )
        dataset = RandomDatasetCreator(low=-1, high=1)

        # Use quantized dataset and allow single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset,
            comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    @staticmethod
    def assert_not_delegated(model, input_shape, use_qat):
        delegated_ep = to_quantized_edge_program(
            model,
            input_shape,
            use_qat=use_qat,
        ).exported_program()

        # Make sure the `convolution` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [Convolution])

    @pytest.mark.parametrize(
        "input_shape, out_channels, is_qat",
        [
            pytest.param(
                ins := (1, 8, 16, 24),
                oc := 8,
                qat := True,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 8, 16, 24),
                oc := 8,
                qat := False,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                oc := 16,
                qat := True,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                oc := 16,
                qat := False,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                oc := 32,
                qat := True,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                oc := 32,
                qat := False,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 8, 32, 64),
                oc := 16,
                qat := True,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 8, 32, 64),
                oc := 16,
                qat := False,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                oc := 24,
                qat := True,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                oc := 24,
                qat := False,
                id=f"qat={qat}, basic inference: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__basic_nsys_inference(
        self, input_shape, out_channels, is_qat, request, mocker
    ):
        in_channels = input_shape[1]
        model = Conv2dModule(in_channels=in_channels, out_channels=out_channels)

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, is_qat",
        [
            pytest.param(
                ins := (1, 8, 16, 24),
                qat := True,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 8, 16, 24),
                qat := False,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                qat := True,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                qat := False,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                qat := True,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                qat := False,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 16, 32, 64),
                qat := True,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 16, 32, 64),
                qat := False,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                qat := True,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                qat := False,
                id=f"qat={qat}, basic inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
        ],
    )
    def test__depthwise(self, input_shape, is_qat, request, mocker):
        out_channels = input_shape[1]
        group = input_shape[1]
        model = Conv2dModule(
            in_channels=input_shape[1], out_channels=out_channels, group=group
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, is_qat",
        [
            pytest.param(
                ins := (1, 3, 7, 14),
                oc := 3,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 3, 7, 14),
                oc := 3,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                oc := 7,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                oc := 7,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                oc := 4,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                oc := 4,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                oc := 1,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                oc := 1,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                oc := 10,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                oc := 10,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                oc := 27,
                qat := True,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                oc := 27,
                qat := False,
                id=f"qat={qat}, unusual shape inference: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__unusual_shapes(self, input_shape, out_channels, is_qat, request, mocker):
        model = Conv2dModule(in_channels=input_shape[1], out_channels=out_channels)

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, is_qat",
        [
            pytest.param(
                ins := (1, 3, 7, 14),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 3, 7, 14),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                qat := True,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                qat := False,
                id=f"qat={qat}, unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
        ],
    )
    def test__depthwise__unusual_shapes(self, input_shape, is_qat, request, mocker):
        out_channels = input_shape[1]
        group = input_shape[1]

        model = Conv2dModule(
            in_channels=input_shape[1], out_channels=out_channels, group=group
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, is_qat",
        [
            pytest.param(
                ins := (21, 4, 7),
                oc := 45,
                qat := True,
                id=f"qat={qat}, `conv2d` implicit batch: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (21, 4, 7),
                oc := 45,
                qat := False,
                id=f"qat={qat}, `conv2d` implicit batch: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__implicit_batch(self, input_shape, out_channels, is_qat, mocker, request):
        in_channels = input_shape[0]

        model = Conv2dModule(in_channels=in_channels, out_channels=out_channels)

        # `view_copy` is inserted to convert to explicit batch
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Convolution: 1, ViewCopy: 2},
            expected_non_delegated_ops={},
        )
        dataset = RandomDatasetCreator(low=-256, high=256)
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset,
            comparator,
            use_qat=is_qat,
        )

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, is_qat",
        [
            pytest.param(
                ins := (2, 3, 1, 4100),
                oc := 7,
                ks := (1, 4096),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 3, 1, 4100),
                oc := 7,
                ks := (1, 4096),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                oc := 9,
                ks := (4096, 1),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                oc := 9,
                ks := (4096, 1),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                oc := 5,
                ks := 3,
                s := (1, 4096),
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of stride width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                oc := 5,
                ks := 3,
                s := (1, 4096),
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of stride width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 3, 8500, 3),
                oc := 11,
                ks := 3,
                s := (4096, 1),
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of stride height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 3, 8500, 3),
                oc := 11,
                ks := 3,
                s := (4096, 1),
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of stride height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 3, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4096),
                qat := True,
                id=f"qat={qat}, bounds of dilation width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 3, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4096),
                qat := False,
                id=f"qat={qat}, bounds of dilation width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4096, 1),
                qat := True,
                id=f"qat={qat}, bounds of dilation height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4096, 1),
                qat := False,
                id=f"qat={qat}, bounds of dilation height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                oc := 13,
                ks := (32, 24),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel_h * kernel_w * input_channels: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
                marks=pytest.mark.xfail(
                    reason="AIR-14679",
                    strict=True,
                ),
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                oc := 13,
                ks := (32, 24),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel_h * kernel_w * input_channels: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
                marks=pytest.mark.xfail(
                    reason="AIR-14679",
                    strict=True,
                ),
            ),
        ],
    )
    def test__big(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        is_qat,
        request,
        mocker,
    ):
        model = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation, is_qat",
        [
            pytest.param(
                ins := (2, 3, 1, 4100),
                ks := (1, 4096),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 3, 1, 4100),
                ks := (1, 4096),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                ks := (4096, 1),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                ks := (4096, 1),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 3, 3, 8500),
                ks := 3,
                s := (1, 4096),
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of stride width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 3, 3, 8500),
                ks := 3,
                s := (1, 4096),
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of stride width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                ks := 3,
                s := (4096, 1),
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of stride height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                ks := 3,
                s := (4096, 1),
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of stride height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                ks := 3,
                s := 1,
                d := (1, 4096),
                qat := True,
                id=f"qat={qat}, bounds of dilation width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                ks := 3,
                s := 1,
                d := (1, 4096),
                qat := False,
                id=f"qat={qat}, bounds of dilation width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 8500, 3),
                ks := 3,
                s := 1,
                d := (4096, 1),
                qat := True,
                id=f"qat={qat}, bounds of dilation height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 8500, 3),
                ks := 3,
                s := 1,
                d := (4096, 1),
                qat := False,
                id=f"qat={qat}, bounds of dilation height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                ks := (32, 24),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, bounds of kernel_h * kernel_w * input_channels: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                ks := (32, 24),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, bounds of kernel_h * kernel_w * input_channels: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
        ],
    )
    def test__depthwise__big(
        self, input_shape, kernel_size, stride, dilation, is_qat, request, mocker
    ):
        out_channels = input_shape[1]
        group = input_shape[1]

        model = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            group=group,
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, padding, bias, is_qat",
        [
            pytest.param(
                ins := (1, 8, 32, 32),
                oc := 7,
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (1, 8, 32, 32),
                oc := 7,
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 7, 31, 17),
                oc := 9,
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 7, 31, 17),
                oc := 9,
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                oc := 11,
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                oc := 11,
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                oc := 13,
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                oc := 13,
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                oc := 5,
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                oc := 5,
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                oc := 7,
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                oc := 7,
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                oc := 9,
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                oc := 9,
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
        ],
    )
    def test__non_default_params(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        bias,
        is_qat,
        request,
        mocker,
    ):
        model = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation, padding, bias, is_qat",
        [
            pytest.param(
                ins := (1, 8, 32, 32),
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (1, 8, 32, 32),
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 31, 17),
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 31, 17),
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                qat := True,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                qat := False,
                id=f"qat={qat}, some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
        ],
    )
    def test__depthwise__non_default_params(
        self,
        input_shape,
        kernel_size,
        stride,
        dilation,
        padding,
        bias,
        is_qat,
        request,
        mocker,
    ):
        out_channels = input_shape[1]
        group = input_shape[1]

        model = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            group=group,
        )

        self.assert_delegated_and_correct(model, input_shape, mocker, request, is_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, is_qat",
        [
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 7,
                ks := (4097, 1),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 7,
                ks := (4097, 1),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 9,
                ks := (1, 4097),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 9,
                ks := (1, 4097),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 11,
                ks := 3,
                s := (4097, 1),
                d := 1,
                qat := True,
                id=f"qat={qat}, stride height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 11,
                ks := 3,
                s := (4097, 1),
                d := 1,
                qat := False,
                id=f"qat={qat}, stride height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 5,
                ks := 3,
                s := (1, 4097),
                d := 1,
                qat := True,
                id=f"qat={qat}, stride width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 5,
                ks := 3,
                s := (1, 4097),
                d := 1,
                qat := False,
                id=f"qat={qat}, stride width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4097, 1),
                qat := True,
                id=f"qat={qat}, dilation height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4097, 1),
                qat := False,
                id=f"qat={qat}, dilation height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4097),
                qat := True,
                id=f"qat={qat}, dilation width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4097),
                qat := False,
                id=f"qat={qat}, dilation width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                oc := 11,
                ks := (41, 15),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel_h * kernel_w * input_channels too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                oc := 11,
                ks := (41, 15),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel_h * kernel_w * input_channels too big: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
        ],
    )
    def test__non_delegation(
        self, input_shape, out_channels, kernel_size, stride, dilation, is_qat
    ):
        in_channels = input_shape[1]

        model = Conv2dModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.assert_not_delegated(model, input_shape, is_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation, is_qat",
        [
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := (4097, 1),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := (4097, 1),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := (1, 4097),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := (1, 4097),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := 3,
                s := (4097, 1),
                d := 1,
                qat := True,
                id=f"qat={qat}, stride height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := 3,
                s := (4097, 1),
                d := 1,
                qat := False,
                id=f"qat={qat}, stride height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := 3,
                s := (1, 4097),
                d := 1,
                qat := True,
                id=f"qat={qat}, stride width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := 3,
                s := (1, 4097),
                d := 1,
                qat := False,
                id=f"qat={qat}, stride width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                ks := 3,
                s := 1,
                d := (4097, 1),
                qat := True,
                id=f"qat={qat}, dilation height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                ks := 3,
                s := 1,
                d := (4097, 1),
                qat := False,
                id=f"qat={qat}, dilation height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                ks := 3,
                s := 1,
                d := (1, 4097),
                qat := True,
                id=f"qat={qat}, dilation width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                ks := 3,
                s := 1,
                d := (1, 4097),
                qat := False,
                id=f"qat={qat}, dilation width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                ks := (41, 15),
                s := 1,
                d := 1,
                qat := True,
                id=f"qat={qat}, kernel_h * kernel_w * input_channels too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                ks := (41, 15),
                s := 1,
                d := 1,
                qat := False,
                id=f"qat={qat}, kernel_h * kernel_w * input_channels too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
        ],
    )
    def test__non_delegation_depthwise(
        self, input_shape, kernel_size, stride, dilation, is_qat
    ):
        out_channels = input_shape[1]
        group = input_shape[1]

        model = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            group=group,
        )

        self.assert_not_delegated(model, input_shape, is_qat)
