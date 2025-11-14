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


def all_flows() -> dict[str, TestFlow]:
    flows = []

    from executorch.backends.test.suite.flows.portable import PORTABLE_TEST_FLOW

    flows += [
        PORTABLE_TEST_FLOW,
    ]

    try:
        from executorch.backends.test.suite.flows.xnnpack import (
            XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW,
            XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
            XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW,
            XNNPACK_TEST_FLOW,
        )

        flows += [
            XNNPACK_TEST_FLOW,
            XNNPACK_DYNAMIC_INT8_PER_CHANNEL_TEST_FLOW,
            XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
            XNNPACK_STATIC_INT8_PER_TENSOR_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping XNNPACK flow registration: {e}")

    try:
        from executorch.backends.test.suite.flows.coreml import (
            COREML_STATIC_INT8_TEST_FLOW,
            COREML_TEST_FLOW,
        )

        flows += [
            COREML_TEST_FLOW,
            COREML_STATIC_INT8_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping Core ML flow registration: {e}")

    try:
        from executorch.backends.test.suite.flows.vulkan import (
            VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
            VULKAN_TEST_FLOW,
        )

        flows += [
            VULKAN_TEST_FLOW,
            VULKAN_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping Vulkan flow registration: {e}")

    try:
        from executorch.backends.test.suite.flows.qualcomm import (
            QNN_16A16W_TEST_FLOW,
            QNN_16A4W_BLOCK_TEST_FLOW,
            QNN_16A4W_TEST_FLOW,
            QNN_16A8W_TEST_FLOW,
            QNN_8A8W_TEST_FLOW,
            QNN_TEST_FLOW,
        )

        flows += [
            QNN_TEST_FLOW,
            QNN_16A16W_TEST_FLOW,
            QNN_16A8W_TEST_FLOW,
            QNN_16A4W_TEST_FLOW,
            QNN_16A4W_BLOCK_TEST_FLOW,
            QNN_8A8W_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping QNN flow registration: {e}")

    try:
        from executorch.backends.test.suite.flows.arm import (
            ARM_ETHOS_U55_FLOW,
            ARM_ETHOS_U85_FLOW,
            ARM_TOSA_FP_FLOW,
            ARM_TOSA_INT_FLOW,
            ARM_VGF_FP_FLOW,
            ARM_VGF_INT_FLOW,
        )

        flows += [
            ARM_TOSA_FP_FLOW,
            ARM_TOSA_INT_FLOW,
            ARM_ETHOS_U55_FLOW,
            ARM_ETHOS_U85_FLOW,
            ARM_VGF_FP_FLOW,
            ARM_VGF_INT_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping ARM flow registration: {e}")

    return {f.name: f for f in flows if f is not None}
