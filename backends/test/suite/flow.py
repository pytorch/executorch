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

    quantize: bool = field(default=False)
    """ Whether to tester should run the quantize stage on the model. """

    quantize_stage_factory: Callable[..., Quantize] | None = None
    """ A factory function which instantiates a Quantize stage. Can be None to use the tester's default. """


def all_flows() -> dict[str, TestFlow]:
    flows = []

    try:
        from executorch.backends.test.suite.flows.xnnpack import (
            XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
            XNNPACK_TEST_FLOW,
        )

        flows += [
            XNNPACK_TEST_FLOW,
            XNNPACK_STATIC_INT8_PER_CHANNEL_TEST_FLOW,
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
        from executorch.backends.test.suite.flows.vulkan import VULKAN_TEST_FLOW

        flows += [
            VULKAN_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping Vulkan flow registration: {e}")

    try:
        from executorch.backends.test.suite.flows.qualcomm import QUALCOMM_TEST_FLOW

        flows += [
            QUALCOMM_TEST_FLOW,
        ]
    except Exception as e:
        logger.info(f"Skipping Qualcomm flow registration: {e}")

    return {f.name: f for f in flows if f is not None}
