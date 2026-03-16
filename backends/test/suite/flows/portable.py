import logging

from executorch.backends.test.harness import Tester
from executorch.backends.test.suite.flow import TestFlow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_portable_flow() -> TestFlow:
    return TestFlow(
        "portable",
        backend="portable",
        tester_factory=Tester,
        is_delegated=False,
    )


PORTABLE_TEST_FLOW = _create_portable_flow()
