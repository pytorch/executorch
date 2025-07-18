import logging

from dataclasses import dataclass
from math import log
from typing import Callable, Sequence

from executorch.backends.test.harness import Tester

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

    tester_factory: Callable[[], Tester]
    """ A factory function that returns a Tester instance for this lowering flow. """


def create_xnnpack_flow() -> TestFlow | None:
    try:
        from executorch.backends.xnnpack.test.tester import Tester as XnnpackTester

        return TestFlow(
            name="xnnpack",
            backend="xnnpack",
            tester_factory=XnnpackTester,
        )
    except Exception:
        logger.info("Skipping XNNPACK flow registration due to import failure.")
        return None


def create_coreml_flow() -> TestFlow | None:
    try:
        from executorch.backends.apple.coreml.test.tester import CoreMLTester

        return TestFlow(
            name="coreml",
            backend="coreml",
            tester_factory=CoreMLTester,
        )
    except Exception:
        logger.info("Skipping Core ML flow registration due to import failure.")
        return None


def all_flows() -> Sequence[TestFlow]:
    flows = [
        create_xnnpack_flow(),
        create_coreml_flow(),
    ]
    return [f for f in flows if f is not None]
