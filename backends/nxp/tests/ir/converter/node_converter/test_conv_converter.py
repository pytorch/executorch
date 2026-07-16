# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dModule, Conv2dTransposedModule
from executorch.backends.nxp.tests.nsys_testing import (
    AllCloseOutputComparator,
    lower_run_compare,
    ReferenceModel,
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


def assert_delegated_and_correct(
    model,
    input_shape,
    mocker,
    request,
    use_qat,
    exp_delegated_ops=None,
    exp_non_delegated_ops=None,
    et_ref_model=ReferenceModel.QUANTIZED_EXECUTORCH_CPP,
):
    if exp_delegated_ops is None:
        exp_delegated_ops = {Convolution: 1}
    if exp_non_delegated_ops is None:
        exp_non_delegated_ops = {}

    graph_verifier = DetailedGraphVerifier(
        mocker,
        expected_delegated_ops=exp_delegated_ops,
        expected_non_delegated_ops=exp_non_delegated_ops,
    )
    dataset = RandomDatasetCreator(low=-1.0, high=1.0)

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
        reference_model=et_ref_model,
    )


def assert_not_delegated(model, input_shape, use_qat):
    delegated_ep = to_quantized_edge_program(
        model,
        input_shape,
        use_qat=use_qat,
    ).exported_program()

    # Make sure the `convolution` was NOT delegated.
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [Convolution])


def _conv_id(ins, oc, ks=3, s=2, d=1, p=0, op=0, b=True, g=1):
    return (
        f"ins={ins}, "
        f"oc={oc}, "
        f"ks={ks}, "
        f"s={s}, "
        f"d={d}, "
        f"p={p}, "
        f"op={op}, "
        f"b={b}, "
        f"g={g}"
    )


class TestTrConv:
    @pytest.mark.parametrize(
        "input_shape, out_channels",
        [
            pytest.param(
                ins := (1, 8, 16, 24),
                oc := 8,
                id=f"basic inference: {_conv_id(ins, oc)}",
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                oc := 16,
                id=f"basic inference: {_conv_id(ins, oc)}",
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                oc := 32,
                id=f"basic inference: {_conv_id(ins, oc)}",
                marks=pytest.mark.xfail(reason="AIR-14853", strict=True),
            ),
            pytest.param(
                ins := (1, 8, 32, 64),
                oc := 16,
                id=f"basic inference: {_conv_id(ins, oc)}",
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                oc := 24,
                id=f"basic inference: {_conv_id(ins, oc)}",
            ),
        ],
    )
    def test__tr_basic(self, input_shape, out_channels, use_qat, request, mocker):
        in_channels = input_shape[1]
        model = Conv2dTransposedModule(
            in_channels=in_channels, out_channels=out_channels
        )

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, use_et_ref_model",
        [
            pytest.param(
                ins := (1, 3, 7, 14),
                oc := 3,
                use_et_ref_model := True,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                oc := 7,
                use_et_ref_model := True,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                oc := 4,
                use_et_ref_model := True,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 9, 9, 13),
                oc := 1,
                use_et_ref_model := False,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                oc := 10,
                use_et_ref_model := True,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                oc := 27,
                use_et_ref_model := True,
                id=f"ET reference model used: {use_et_ref_model}, unusual shape inference: "
                + _conv_id(ins, oc),
            ),
        ],
    )
    def test__tr_unusual(
        self, input_shape, out_channels, use_et_ref_model, use_qat, request, mocker
    ):
        in_channels = input_shape[1]
        model = Conv2dTransposedModule(
            in_channels=in_channels, out_channels=out_channels
        )

        # Running `conv_transpose2d` with `output_channels = 1` produces errors in Executorch. The issue has been reported:
        # https://github.com/pytorch/executorch/issues/20804
        ref_model = (
            ReferenceModel.QUANTIZED_EXECUTORCH_CPP
            if use_et_ref_model
            else ReferenceModel.QUANTIZED_EDGE_PYTHON
        )

        assert_delegated_and_correct(
            model,
            input_shape,
            mocker,
            request,
            use_qat,
            exp_delegated_ops={Convolution: 1},
            exp_non_delegated_ops={},
            et_ref_model=ref_model,
        )

    @pytest.mark.parametrize(
        "input_shape, out_channels",
        [
            pytest.param(
                ins := (21, 4, 7),
                oc := 45,
                id=f"`conv2d_transpose` implicit batch: {_conv_id(ins, oc)}",
            ),
        ],
    )
    def test__tr_impl_b(self, input_shape, out_channels, use_qat, mocker, request):
        in_channels = input_shape[0]

        model = Conv2dTransposedModule(
            in_channels=in_channels, out_channels=out_channels
        )

        # `view_copy` is inserted to convert to explicit batch
        assert_delegated_and_correct(
            model,
            input_shape,
            mocker,
            request,
            use_qat,
            exp_delegated_ops={Convolution: 1, ViewCopy: 2},
            exp_non_delegated_ops={},
        )

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, padding",
        [
            pytest.param(
                ins := (2, 3, 1, 8500),
                oc := 7,
                ks := (1, 4096),
                s := 1,
                d := 1,
                p := 0,
                id=f"bounds of kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
                marks=pytest.mark.xfail(reason="AIR-14853", strict=True),
            ),
            pytest.param(
                ins := (3, 3, 8500, 1),
                oc := 9,
                ks := (4096, 1),
                s := 1,
                d := 1,
                p := 0,
                id=f"bounds of kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
                marks=pytest.mark.xfail(reason="AIR-14853", strict=True),
            ),
            pytest.param(
                ins := (3, 3, 5, 7),
                oc := 9,
                ks := (2, 1),
                s := (2, 1),
                d := 1,
                p := 0,
                id=f"bounds of stride height - kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 3, 5, 7),
                oc := 9,
                ks := (1, 2),
                s := (1, 2),
                d := 1,
                p := 0,
                id=f"bounds of stride width - kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 3, 5, 7),
                oc := 9,
                ks := (3, 3),
                s := 1,
                d := 1,
                p := (2, 1),
                id=f"bounds of padding height - kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 3, 5, 7),
                oc := 9,
                ks := (3, 3),
                s := 1,
                d := 1,
                p := (1, 2),
                id=f"bounds of padding width - kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (1, 20, 16, 24),
                oc := 2,
                ks := (16, 24),
                s := 1,
                d := 1,
                p := 1,
                id=f"(almost) bounds of kernel_h * kernel_w * round_ceil(input_channels, num_macs): {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
        ],
    )
    def test__tr_big(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        use_qat,
        request,
        mocker,
    ):
        model = Conv2dTransposedModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, padding, output_padding, bias",
        [
            pytest.param(
                ins := (1, 8, 32, 32),
                oc := 7,
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                op := (0, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14852", strict=True),
            ),
            pytest.param(
                ins := (2, 7, 31, 17),
                oc := 9,
                ks := (7, 7),
                s := (2, 2),
                d := (6, 5),
                p := (5, 4),
                op := (2, 1),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14853", strict=True),
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                oc := 11,
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                op := (1, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14852", strict=True),
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                oc := 13,
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 4),
                op := (1, 1),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                oc := 5,
                ks := (3, 3),
                s := (2, 2),
                d := (3, 3),
                p := (2, 2),
                op := (2, 2),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14852", strict=True),
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                oc := 7,
                ks := (5, 5),
                s := (1, 2),
                d := (1, 3),
                p := (2, 4),
                op := (0, 2),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14852", strict=True),
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                oc := 9,
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                op := (1, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b, op=op)}",
                marks=pytest.mark.xfail(reason="AIR-14852", strict=True),
            ),
        ],
    )
    def test__tr_misc_arg(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        output_padding,
        bias,
        use_qat,
        request,
        mocker,
    ):
        model = Conv2dTransposedModule(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, padding, groups",
        [
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 7,
                ks := (4097, 1),
                s := 1,
                p := 0,
                g := 1,
                id=f"kernel height too big: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 9,
                ks := (1, 4097),
                s := 1,
                p := 0,
                g := 1,
                id=f"kernel width too big: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 9, 11),
                oc := 11,
                ks := 1,
                s := (2, 1),
                p := 0,
                g := 1,
                id=f"stride height > kernel height: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 11),
                oc := 5,
                ks := 1,
                s := (1, 2),
                p := 0,
                g := 1,
                id=f"stride width > kernel width: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 11),
                oc := 7,
                ks := 3,
                s := (3, 1),
                p := 0,
                g := 1,
                id=f"stride height too big: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 11),
                oc := 7,
                ks := 3,
                s := (1, 3),
                p := 0,
                g := 1,
                id=f"stride width too big: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 9, 11),
                oc := 7,
                ks := 3,
                s := 1,
                p := (3, 1),
                g := 1,
                id=f"padding height >= kernel height: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 9, 11),
                oc := 7,
                ks := 3,
                s := 1,
                p := (1, 3),
                g := 1,
                id=f"padding width >= kernel width: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                oc := 11,
                ks := (41, 15),
                s := 1,
                p := 0,
                g := 1,
                id=f"kernel_h * kernel_w * round_ceil(input_channels, num_macs) too big: {_conv_id(ins, oc, ks=ks, s=s, p=p)}",
            ),
            pytest.param(
                ins := (3, 9, 11, 13),
                oc := 3,
                ks := 3,
                s := 1,
                p := 0,
                g := 3,
                id=f"groups > 1: {_conv_id(ins, oc, ks=ks, s=s, p=p, g=g)}",
            ),
        ],
    )
    def test__tr_no_deleg(
        self, input_shape, out_channels, kernel_size, stride, padding, groups, use_qat
    ):
        in_channels = input_shape[1]

        model = Conv2dTransposedModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        assert_not_delegated(model, input_shape, use_qat)


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

    @pytest.mark.parametrize(
        "input_shape, out_channels",
        [
            pytest.param(
                ins := (1, 8, 16, 24),
                oc := 8,
                id="basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                oc := 16,
                id="basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                oc := 32,
                id="basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 8, 32, 64),
                oc := 16,
                id="basic inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                oc := 24,
                id="basic inference: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__fwd_basic(self, input_shape, out_channels, use_qat, request, mocker):
        in_channels = input_shape[1]
        model = Conv2dModule(in_channels=in_channels, out_channels=out_channels)

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param(
                ins := (1, 8, 16, 24),
                id="basic inference, depthwise: " + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (8, 16, 8, 32),
                id="basic inference, depthwise: " + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (16, 8, 32, 64),
                id="basic inference, depthwise: " + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 16, 32, 64),
                id="basic inference, depthwise: " + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 32, 48, 8),
                id="basic inference, depthwise: " + _conv_id(ins, ins[1], g=ins[1]),
            ),
        ],
    )
    def test__d_fwd_basic(self, input_shape, use_qat, request, mocker):
        out_channels = input_shape[1]
        group = input_shape[1]
        model = Conv2dModule(
            in_channels=input_shape[1], out_channels=out_channels, group=group
        )

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels",
        [
            pytest.param(
                ins := (1, 3, 7, 14),
                oc := 3,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                oc := 7,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                oc := 4,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                oc := 1,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                oc := 10,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                oc := 27,
                id="unusual shape inference: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__fwd_unusual(self, input_shape, out_channels, use_qat, request, mocker):
        model = Conv2dModule(in_channels=input_shape[1], out_channels=out_channels)

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param(
                ins := (1, 3, 7, 14),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (2, 3, 13, 27),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (3, 7, 3, 14),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (1, 7, 7, 21),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (7, 7, 7, 7),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
            pytest.param(
                ins := (4, 21, 13, 17),
                id="unusual shape inference, depthwise: "
                + _conv_id(ins, ins[1], g=ins[1]),
            ),
        ],
    )
    def test__d_fwd_unusual(self, input_shape, use_qat, request, mocker):
        out_channels = input_shape[1]
        group = input_shape[1]

        model = Conv2dModule(
            in_channels=input_shape[1], out_channels=out_channels, group=group
        )

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels",
        [
            pytest.param(
                ins := (21, 4, 7),
                oc := 45,
                id="`conv2d` implicit batch: " + _conv_id(ins, oc),
            ),
        ],
    )
    def test__fwd_impl_b(self, input_shape, out_channels, use_qat, mocker, request):
        in_channels = input_shape[0]

        model = Conv2dModule(in_channels=in_channels, out_channels=out_channels)

        # `view_copy` is inserted to convert to explicit batch
        assert_delegated_and_correct(
            model,
            input_shape,
            mocker,
            request,
            use_qat,
            exp_delegated_ops={Convolution: 1, ViewCopy: 2},
            exp_non_delegated_ops={},
        )

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation",
        [
            pytest.param(
                ins := (2, 3, 1, 4100),
                oc := 7,
                ks := (1, 4096),
                s := 1,
                d := 1,
                id=f"bounds of kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                oc := 9,
                ks := (4096, 1),
                s := 1,
                d := 1,
                id=f"bounds of kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                oc := 5,
                ks := 3,
                s := (1, 4096),
                d := 1,
                id=f"bounds of stride width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 3, 8500, 3),
                oc := 11,
                ks := 3,
                s := (4096, 1),
                d := 1,
                id=f"bounds of stride height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (3, 3, 3, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4096),
                id=f"bounds of dilation width: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4096, 1),
                id=f"bounds of dilation height: {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                oc := 13,
                ks := (32, 24),
                s := 1,
                d := 1,
                id=f"bounds of kernel_h * kernel_w * round_ceil(input_channels, num_macs): {_conv_id(ins, oc, ks=ks, s=s, d=d)}",
                marks=pytest.mark.xfail(
                    reason="AIR-14679",
                    strict=True,
                ),
            ),
        ],
    )
    def test__fwd_big(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        use_qat,
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

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation",
        [
            pytest.param(
                ins := (2, 3, 1, 4100),
                ks := (1, 4096),
                s := 1,
                d := 1,
                id=f"bounds of kernel width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 4100, 1),
                ks := (4096, 1),
                s := 1,
                d := 1,
                id=f"bounds of kernel height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 3, 3, 8500),
                ks := 3,
                s := (1, 4096),
                d := 1,
                id=f"bounds of stride width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 8500, 3),
                ks := 3,
                s := (4096, 1),
                d := 1,
                id=f"bounds of stride height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 3, 3, 8500),
                ks := 3,
                s := 1,
                d := (1, 4096),
                id=f"bounds of dilation width: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 3, 8500, 3),
                ks := 3,
                s := 1,
                d := (4096, 1),
                id=f"bounds of dilation height: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 80, 35, 34),
                ks := (32, 24),
                s := 1,
                d := 1,
                id=f"bounds of kernel_h * kernel_w * round_ceil(input_channels, num_macs): {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1])}",
            ),
        ],
    )
    def test__d_fwd_big(
        self, input_shape, kernel_size, stride, dilation, use_qat, request, mocker
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

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, padding, bias",
        [
            pytest.param(
                ins := (1, 8, 32, 32),
                oc := 7,
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 7, 31, 17),
                oc := 9,
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                oc := 11,
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                oc := 13,
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                oc := 5,
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                oc := 7,
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                oc := 9,
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p, b=b)}",
            ),
        ],
    )
    def test__fwd_misc_arg(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        bias,
        use_qat,
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

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation, padding, bias",
        [
            pytest.param(
                ins := (1, 8, 32, 32),
                ks := (5, 3),
                s := (2, 1),
                d := (1, 2),
                p := (2, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 7, 31, 17),
                ks := (7, 7),
                s := (3, 2),
                d := (2, 1),
                p := (3, 3),
                b := False,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (2, 12, 28, 28),
                ks := (3, 5),
                s := (2, 2),
                d := (2, 2),
                p := (1, 2),
                b := True,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 2, 40, 20),
                ks := (1, 5),
                s := (1, 2),
                d := (3, 1),
                p := (0, 2),
                b := False,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (4, 6, 30, 30),
                ks := (3, 3),
                s := (2, 2),
                d := (1, 1),
                p := (2, 2),
                b := True,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (3, 12, 7, 7),
                ks := (5, 5),
                s := (1, 3),
                d := (1, 2),
                p := (2, 4),
                b := False,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
            pytest.param(
                ins := (1, 4, 15, 15),
                ks := (2, 2),
                s := (2, 2),
                d := (2, 2),
                p := (1, 1),
                b := True,
                id=f"some params not default: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, p=p, b=b, g=ins[1])}",
            ),
        ],
    )
    def test__d_fwd_misc_arg(
        self,
        input_shape,
        kernel_size,
        stride,
        dilation,
        padding,
        bias,
        use_qat,
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

        assert_delegated_and_correct(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "input_shape, out_channels, kernel_size, stride, dilation, padding",
        [
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 7,
                ks := (4097, 1),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 9,
                ks := (1, 4097),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                oc := 11,
                ks := 3,
                s := (4097, 1),
                d := 1,
                p := 0,
                id=f"stride height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                oc := 5,
                ks := 3,
                s := (1, 4097),
                d := 1,
                p := 0,
                id=f"stride width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            # The following two cases are delegable for now, the discussion about
            # the validity of such cases is being discussed with Neutron team.
            # See more in `convolution_converter`.
            # pytest.param(
            #     ins := (3, 7, 13, 11),
            #     oc := 5,
            #     ks := (3, 1),
            #     s := 1,
            #     d := 1,
            #     p := (3, 1),
            #     id=f"padding height >= kernel height: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            # ),
            # pytest.param(
            #     ins := (3, 7, 13, 11),
            #     oc := 5,
            #     ks := (1, 3),
            #     s := 1,
            #     d := 1,
            #     p := (1, 3),
            #     id=f"padding width >= kernel width: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            # ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                oc := 7,
                ks := 3,
                s := 1,
                d := (4097, 1),
                p := 0,
                id=f"dilation height too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                oc := 9,
                ks := 3,
                s := 1,
                d := (1, 4097),
                p := 0,
                id=f"dilation width too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                oc := 11,
                ks := (41, 15),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel_h * kernel_w * round_ceil(input_channels, num_macs) too big: {_conv_id(ins, oc, ks=ks, s=s, d=d, p=p)}",
            ),
        ],
    )
    def test__fwd_no_deleg(
        self, input_shape, out_channels, kernel_size, stride, dilation, padding, use_qat
    ):
        in_channels = input_shape[1]

        model = Conv2dModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

        assert_not_delegated(model, input_shape, use_qat)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, dilation, padding",
        [
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := (4097, 1),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := (1, 4097),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 5000, 11),
                ks := 3,
                s := (4097, 1),
                d := 1,
                p := 0,
                id=f"stride height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 5000),
                ks := 3,
                s := (1, 4097),
                d := 1,
                p := 0,
                id=f"stride width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            # The following two cases are delegable for now, the discussion about
            # the validity of such cases is being discussed with Neutron team.
            # See more in `convolution_converter`.
            # pytest.param(
            #     ins := (3, 7, 13, 11),
            #     ks := (3, 1),
            #     s := 1,
            #     d := 1,
            #     p := (3, 1),
            #     id=f"padding height >= kernel height, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            # ),
            # pytest.param(
            #     ins := (3, 7, 13, 11),
            #     ks := (1, 3),
            #     s := 1,
            #     d := 1,
            #     p := (1, 3),
            #     id=f"padding width >= kernel width, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            # ),
            pytest.param(
                ins := (3, 7, 8500, 11),
                ks := 3,
                s := 1,
                d := (4097, 1),
                p := 0,
                id=f"dilation height too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            pytest.param(
                ins := (3, 7, 13, 8500),
                ks := 3,
                s := 1,
                d := (1, 4097),
                p := 0,
                id=f"dilation width too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
            pytest.param(
                ins := (3, 113, 123, 133),
                ks := (41, 15),
                s := 1,
                d := 1,
                p := 0,
                id=f"kernel_h * kernel_w * round_ceil(input_channels, num_macs) too big, depthwise: {_conv_id(ins, ins[1], ks=ks, s=s, d=d, g=ins[1], p=p)}",
            ),
        ],
    )
    def test__d_fwd_no_deleg(
        self, input_shape, kernel_size, stride, dilation, padding, use_qat
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
            padding=padding,
        )

        assert_not_delegated(model, input_shape, use_qat)
