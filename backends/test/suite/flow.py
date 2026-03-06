# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Callable

from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import Quantize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TestFlow:
    """
    A lowering flow to test. This typically corresponds to a combination of a backend and
    a lowering recipe.
    """

    name: str
    """ The name of the lowering flow. """

    backend: str
    """ The name of the target backend. """

    tester_factory: Callable[..., Tester]
    """ A factory function that returns a Tester instance for this lowering flow. """

    quantize: bool = False
    """ Whether to tester should run the quantize stage on the model. """

    quantize_stage_factory: Callable[..., Quantize] | None = None
    """ A factory function which instantiates a Quantize stage. Can be None to use the tester's default. """

    is_delegated: bool = True
    """ Indicates whether the flow is expected to generate CALL_DELEGATE nodes. """

    skip_patterns: list[str] = field(default_factory=lambda: [])
    """ Tests with names containing any substrings in this list are skipped. """

    supports_serialize: bool = True
    """ True if the test flow supports the Serialize stage. """

    def should_skip_test(self, test_name: str) -> bool:
        return any(pattern in test_name for pattern in self.skip_patterns)

    def __str__(self):
        return self.name


def _try_import_flows(
    module_path: str, flow_names: list[str], backend_name: str
) -> list[TestFlow]:
    """
    Attempt to import test flows from a module.

    Args:
        module_path: The full module path to import from.
        flow_names: List of flow variable names to import from the module.
        backend_name: Human-readable name for logging on failure.

    Returns:
        List of imported TestFlow objects, or empty list if import fails.
    """
    try:
        import importlib

        module = importlib.import_module(module_path)
        return [getattr(module, name) for name in flow_names]
    except Exception as e:
        logger.info(f"Skipping {backend_name} flow registration: {e}")
        return []


# Registry of backend flows to import: (module_path, flow_names, backend_name)
_FLOW_REGISTRY: list[tuple[str, list[str], str]] = [
    (
        "executorch.backends.test.suite.flows.xnnpack",
        [
            "XNNPACK_TEST_FLOW",
            "XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW",
            "XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW",
            "XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW",
        ],
        "XNNPACK",
    ),
    (
        "executorch.backends.test.suite.flows.coreml",
        [
            "COREML_TEST_FLOW",
            "COREML_STATIC_INT8_TEST_FLOW",
        ],
        "Core ML",
    ),
    (
        "executorch.backends.test.suite.flows.vulkan",
        [
            "VULKAN_TEST_FLOW",
            "VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW",
        ],
        "Vulkan",
    ),
    (
        "executorch.backends.test.suite.flows.qualcomm",
        [
            "QNN_TEST_FLOW",
            "QNN_16A16W_TEST_FLOW",
            "QNN_16A8W_TEST_FLOW",
            "QNN_16A4W_TEST_FLOW",
            "QNN_16A4W_BLOCK_TEST_FLOW",
            "QNN_8A8W_TEST_FLOW",
        ],
        "QNN",
    ),
    (
        "executorch.backends.test.suite.flows.arm",
        [
            "ARM_TOSA_FP_FLOW",
            "ARM_TOSA_INT_FLOW",
            "ARM_ETHOS_U55_FLOW",
            "ARM_ETHOS_U85_FLOW",
            "ARM_VGF_FP_FLOW",
            "ARM_VGF_INT_FLOW",
        ],
        "ARM",
    ),
    (
        "executorch.backends.test.suite.flows.cuda",
        [
            "CUDA_TEST_FLOW",
        ],
        "CUDA",
    ),
]


def all_flows() -> dict[str, TestFlow]:
    from executorch.backends.test.suite.flows.portable import PORTABLE_TEST_FLOW

    flows = [PORTABLE_TEST_FLOW]

    for module_path, flow_names, backend_name in _FLOW_REGISTRY:
        flows.extend(_try_import_flows(module_path, flow_names, backend_name))

    return {f.name: f for f in flows if f is not None}
