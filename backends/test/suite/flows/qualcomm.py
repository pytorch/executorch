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
    )


QNN_TEST_FLOW = _create_qnn_flow("qnn")
QNN_16A16W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a16w", quantize=True, quant_dtype=QuantDtype.use_8a8w, use_fp16=False
)
QNN_16A8W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a8w", quantize=True, quant_dtype=QuantDtype.use_16a8w, use_fp16=False
)
QNN_16A4W_TEST_FLOW = _create_qnn_flow(
    "qnn_16a4w", quantize=True, quant_dtype=QuantDtype.use_16a4w, use_fp16=False
)
QNN_16A4W_BLOCK_TEST_FLOW = _create_qnn_flow(
    "qnn_16a4w_block",
    quantize=True,
    quant_dtype=QuantDtype.use_8a8w,
    use_fp16=False,
)
QNN_8A8W_TEST_FLOW = _create_qnn_flow(
    "qnn_8a8w", quantize=True, quant_dtype=QuantDtype.use_8a8w, use_fp16=False
)
