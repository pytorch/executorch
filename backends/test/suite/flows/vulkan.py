from executorch.backends.vulkan.test.tester import VulkanTester
from executorch.backends.test.suite.flow import TestFlow

def _create_vulkan_flow(
    name: str, 
    quantize: bool = False, 
) -> TestFlow:
    return TestFlow(
        name,
        backend="vulkan",
        tester_factory=VulkanTester,
        quantize=quantize,
    )

VULKAN_TEST_FLOW = _create_vulkan_flow("vulkan")
