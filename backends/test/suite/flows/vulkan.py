from typing import Callable

from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config as get_symmetric_quantization_config_vulkan,
)
from executorch.backends.vulkan.test.tester import (
    Quantize as VulkanQuantize,
    VulkanTester,
)


def _create_vulkan_flow_base(
    name: str, quantize_stage_factory: Callable[..., Quantize] | None = None
) -> TestFlow:
    return TestFlow(
        name,
        backend="vulkan",
        tester_factory=VulkanTester,
        quantize=quantize_stage_factory is not None,
        quantize_stage_factory=quantize_stage_factory,
    )


def _create_vulkan_flow() -> TestFlow:
    return _create_vulkan_flow_base("vulkan")


def _create_vulkan_static_int8_per_channel_flow() -> TestFlow:
    def create_quantize_stage() -> Quantize:
        qparams = get_symmetric_quantization_config_vulkan()
        return VulkanQuantize(
            quantization_config=qparams,
        )

    return _create_vulkan_flow_base(
        "vulkan_static_int8_per_channel", create_quantize_stage
    )


VULKAN_TEST_FLOW = _create_vulkan_flow()
VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW = _create_vulkan_static_int8_per_channel_flow()
