from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.tests.tester import QualcommTester, Quantize
from executorch.backends.test.suite.flow import TestFlow
from torchao.quantization.pt2e import MovingAverageMinMaxObserver


def _create_qnn_flow(
    name: str,
    quantize: bool = False,
    quant_dtype: QuantDtype | None = None,
    per_channel_conv=True,
    per_channel_linear=False,
    is_qat=False,
    use_fp16=True,
    xfail_patterns: list[str] | None = None,
) -> TestFlow:
    if quantize and quant_dtype is None:
        raise RuntimeError("Quant dtype must be provided when quantize is true.")

    def create_tester(*args, **kwargs) -> QualcommTester:
        kwargs["use_fp16"] = (use_fp16,)
        return QualcommTester(*args, **kwargs)

    def create_quantize_stage() -> Quantize:
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype,
            is_qat=is_qat,
            is_conv_per_channel=per_channel_conv,
            is_linear_per_channel=per_channel_linear,
            act_observer=MovingAverageMinMaxObserver,
        )
        return Quantize(quantizer=quantizer)

    return TestFlow(
        name,
        backend="qualcomm",
        tester_factory=create_tester,
        quantize=quantize,
        quantize_stage_factory=create_quantize_stage if quantize else None,
        xfail_patterns=xfail_patterns or [],
    )


# QNN fails to lower these 3-D convolutions instead of leaving them to the
# portable kernel. Partitioning asserts "DepthWise only supports Conv2d" on the
# depthwise one and raises "'NoneType' object has no attribute 'op'" on the
# replicate-padded one. For the circular-padded one, the QNN graph's inputs and
# outputs do not match the delegated program.
_QNN_CONV3D_LOWERING_XFAILS = [
    "test_conv3d_depthwise",
    "test_conv3d_padding_circular",
    "test_conv3d_padding_replicate",
]
# With 8-bit weights, QNN takes these transposed 3-D convolutions and its output
# does not match.
_QNN_8BIT_WEIGHT_CONV3D_XFAILS = [
    "test_convtranspose3d_depthwise",
    "test_convtranspose3d_groups",
    "test_convtranspose3d_padding",
]


QNN_TEST_FLOW = _create_qnn_flow("qnn")
QNN_16A16W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a16w",
    quantize=True,
    quant_dtype=QuantDtype.use_16a16w,
    use_fp16=False,
    xfail_patterns=_QNN_CONV3D_LOWERING_XFAILS,
)
QNN_16A8W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a8w",
    quantize=True,
    quant_dtype=QuantDtype.use_16a8w,
    use_fp16=False,
    xfail_patterns=_QNN_CONV3D_LOWERING_XFAILS + _QNN_8BIT_WEIGHT_CONV3D_XFAILS,
)
QNN_16A4W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a4w",
    quantize=True,
    quant_dtype=QuantDtype.use_16a4w,
    use_fp16=False,
    # Every 3-D convolution fails in this flow, most of them when QNN generates
    # the context binary.
    xfail_patterns=["test_conv3d", "test_convtranspose3d"],
)
QNN_16A4W_BLOCK_TEST_FLOW = _create_qnn_flow(
    "qnn_16a4w_block",
    quantize=True,
    quant_dtype=QuantDtype.use_8a8w,
    use_fp16=False,
    # This flow still quantizes as 8a8w, not 16a4w block, so it fails the same
    # cases as qnn_8a8w. Recheck the list when the dtype is corrected.
    xfail_patterns=_QNN_CONV3D_LOWERING_XFAILS + _QNN_8BIT_WEIGHT_CONV3D_XFAILS,
)
QNN_8A8W_TEST_FLOW = _create_qnn_flow(
    "qnn_8a8w",
    quantize=True,
    quant_dtype=QuantDtype.use_8a8w,
    use_fp16=False,
    xfail_patterns=_QNN_CONV3D_LOWERING_XFAILS + _QNN_8BIT_WEIGHT_CONV3D_XFAILS,
)
