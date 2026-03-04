import logging

from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.test.tester.samsung_tester import SamsungTester
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_samsung_flow(
    name: str,
    quantize: bool = False,
    quant_dtype: Precision | None = None,
    is_per_channel: bool = True,
    is_qat: bool = False,
) -> TestFlow:
    if quantize and quant_dtype is None:
        raise RuntimeError("Quant dtype must be provided when quantize is true.")

    def create_quantize_stage() -> Quantize:
        quantizer = EnnQuantizer()
        quantizer.setup_quant_params(quant_dtype, is_per_channel, is_qat)
        return Quantize(quantizer=quantizer)

    return TestFlow(
        name,
        backend="samsung",
        tester_factory=SamsungTester,
        quantize=quantize,
        quantize_stage_factory=create_quantize_stage if quantize else None,
        supports_serialize=False,
    )


SAMSUNG_TEST_FLOW = _create_samsung_flow("samsung")

SAMSUNG_A8W8_TEST_FLOW = _create_samsung_flow(
    "samsung_a8w8", quantize=True, quant_dtype=Precision.A8W8
)
