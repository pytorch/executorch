# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from dataclasses import dataclass, field
from typing import Any, Callable

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

    param_skip_reasons: dict[str, dict[Any, str]] = field(default_factory=dict)
    """ Skip tests with a given reason when a pytest parameter matches a given value."""

    supports_serialize: bool = True
    """ True if the test flow supports the Serialize stage. """

    def should_skip_test(
        self, test_name: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        if any(pattern in test_name for pattern in self.skip_patterns):
            return True, f"Skipped by {self.name} skip_patterns"

        if params is None:
            return False, ""

        for param_name, values_to_skip in self.param_skip_reasons.items():
            if param_name not in params:
                continue

            parameter = params[param_name]
            if parameter in values_to_skip:
                return True, values_to_skip[parameter]

        return False, ""

    def __str__(self):
        return self.name


def _register_flow(
    import_fn: Callable[[], list[TestFlow]], backend_name: str
) -> list[TestFlow]:
    try:
        return import_fn()
    except Exception as e:
        logger.info(f"Skipping {backend_name} flow registration: {e}")
        return []


def _load_xnnpack() -> list[TestFlow]:
    from executorch.backends.test.suite.flows.xnnpack import (
        XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW,
        XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
        XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW,
        XNNPACK_TEST_FLOW,
    )

    return [
        XNNPACK_TEST_FLOW,
        XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW,
        XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
        XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW,
    ]


def _load_coreml() -> list[TestFlow]:
    from executorch.backends.test.suite.flows.coreml import (
        COREML_STATIC_INT8_TEST_FLOW,
        COREML_TEST_FLOW,
    )

    return [COREML_TEST_FLOW, COREML_STATIC_INT8_TEST_FLOW]


def _load_vulkan() -> list[TestFlow]:
    from executorch.backends.test.suite.flows.vulkan import (
        VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
        VULKAN_TEST_FLOW,
    )

    return [VULKAN_TEST_FLOW, VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW]


def _load_openvino() -> list[TestFlow]:
    from executorch.backends.test.suite.flows.openvino import (
        OPENVINO_INT8_TEST_FLOW,
        OPENVINO_TEST_FLOW,
    )

    return [OPENVINO_TEST_FLOW, OPENVINO_INT8_TEST_FLOW]


def _load_qnn() -> list[TestFlow]:
    if not os.environ.get("QNN_SDK_ROOT"):
        logger.info("Skipping QNN flow registration: QNN_SDK_ROOT not set")
        return []

    from executorch.backends.test.suite.flows.qualcomm import (
        QNN_16A16W_TEST_FLOW,
        QNN_16A4W_BLOCK_TEST_FLOW,
        QNN_16A4W_TEST_FLOW,
        QNN_16A8W_TEST_FLOW,
        QNN_8A8W_TEST_FLOW,
        QNN_TEST_FLOW,
    )

    return [
        QNN_TEST_FLOW,
        QNN_16A16W_TEST_FLOW,
        QNN_16A8W_TEST_FLOW,
        QNN_16A4W_TEST_FLOW,
        QNN_16A4W_BLOCK_TEST_FLOW,
        QNN_8A8W_TEST_FLOW,
    ]


def _load_arm() -> list[TestFlow]:
    from executorch.backends.test.suite.flows.arm import (
        ARM_ETHOS_U55_FLOW,
        ARM_ETHOS_U85_FLOW,
        ARM_TOSA_FP_FLOW,
        ARM_TOSA_INT_FLOW,
        ARM_VGF_FP_FLOW,
        ARM_VGF_INT_FLOW,
    )

    return [
        ARM_TOSA_FP_FLOW,
        ARM_TOSA_INT_FLOW,
        ARM_ETHOS_U55_FLOW,
        ARM_ETHOS_U85_FLOW,
        ARM_VGF_FP_FLOW,
        ARM_VGF_INT_FLOW,
    ]


def all_flows() -> dict[str, TestFlow]:
    from executorch.backends.test.suite.flows.portable import PORTABLE_TEST_FLOW

    flows = (
        [PORTABLE_TEST_FLOW]
        + _register_flow(_load_xnnpack, "XNNPACK")
        + _register_flow(_load_coreml, "Core ML")
        + _register_flow(_load_vulkan, "Vulkan")
        + _register_flow(_load_openvino, "OpenVINO")
        + _register_flow(_load_qnn, "QNN")
        + _register_flow(_load_arm, "ARM")
    )

    try:
        from executorch.backends.test.suite.flows.mlx import MLX_TEST_FLOW

        flows += [
            MLX_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping MLX flow registration: {e}")

    return {f.name: f for f in flows if f is not None}
