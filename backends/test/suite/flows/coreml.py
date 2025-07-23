import functools
from typing import Any

import coremltools

from executorch.backends.apple.coreml.test.tester import CoreMLTester
from executorch.backends.test.suite.flow import TestFlow


def _create_coreml_flow(
    name: str,
    quantize: bool = False,
    minimum_deployment_target: Any = coremltools.target.iOS15,
) -> TestFlow:
    return TestFlow(
        name,
        backend="coreml",
        tester_factory=functools.partial(
            CoreMLTester, minimum_deployment_target=minimum_deployment_target
        ),
        quantize=quantize,
    )


COREML_TEST_FLOW = _create_coreml_flow("coreml")
COREML_STATIC_INT8_TEST_FLOW = _create_coreml_flow(
    "coreml_static_int8",
    quantize=True,
    minimum_deployment_target=coremltools.target.iOS17,
)
