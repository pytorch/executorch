import logging
from typing import Callable

from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.tester import (
    Quantize as XnnpackQuantize,
    Tester as XnnpackTester,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_xnnpack_flow_base(
    name: str, quantize_stage_factory: Callable[..., Quantize] | None = None
) -> TestFlow:
    return TestFlow(
        name,
        backend="xnnpack",
        tester_factory=XnnpackTester,
        quantize=quantize_stage_factory is not None,
        quantize_stage_factory=quantize_stage_factory,
    )


def _create_xnnpack_flow() -> TestFlow:
    return _create_xnnpack_flow_base("xnnpack")


def _create_xnnpack_dynamic_int8_per_channel_flow() -> TestFlow:
    def create_quantize_stage() -> Quantize:
        qparams = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        return XnnpackQuantize(
            quantization_config=qparams,
        )

    return _create_xnnpack_flow_base(
        "xnnpack_dynamic_int8_per_channel", create_quantize_stage
    )


def _create_xnnpack_static_int8_per_channel_flow() -> TestFlow:
    def create_quantize_stage() -> Quantize:
        qparams = get_symmetric_quantization_config(is_per_channel=True)
        return XnnpackQuantize(
            quantization_config=qparams,
        )

    return _create_xnnpack_flow_base(
        "xnnpack_static_int8_per_channel", create_quantize_stage
    )


def _create_xnnpack_static_int8_per_tensor_flow() -> TestFlow:
    def create_quantize_stage() -> Quantize:
        qparams = get_symmetric_quantization_config(is_per_channel=False)
        return XnnpackQuantize(
            quantization_config=qparams,
        )

    return _create_xnnpack_flow_base(
        "xnnpack_static_int8_per_tensor", create_quantize_stage
    )


XNNPACK_TEST_FLOW = _create_xnnpack_flow()
XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW = (
    _create_xnnpack_dynamic_int8_per_channel_flow()
)
XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW = (
    _create_xnnpack_static_int8_per_channel_flow()
)
XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW = _create_xnnpack_static_int8_per_tensor_flow()
