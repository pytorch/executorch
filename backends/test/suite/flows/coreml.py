import functools
from typing import Any

import coremltools

from executorch.backends.apple.coreml.test.tester import CoreMLTester
from executorch.backends.test.suite.flow import TestFlow


def _create_coreml_flow(
    name: str,
    quantize: bool = False,
    minimum_deployment_target: Any = coremltools.target.iOS15,
    xfail_patterns: list[str] | None = None,
) -> TestFlow:
    return TestFlow(
        name,
        backend="coreml",
        tester_factory=functools.partial(
            CoreMLTester, minimum_deployment_target=minimum_deployment_target
        ),
        quantize=quantize,
        skip_patterns=["test_argmin", "test_argmax"],
        xfail_patterns=xfail_patterns or [],
    )


COREML_TEST_FLOW = _create_coreml_flow(
    "coreml",
    # Core ML fails to compile the model for the reflect padding case, and
    # Core ML lowering of a transposed 3-D convolution with output_padding
    # raises "'NoneType' object is not iterable".
    xfail_patterns=[
        "test_conv3d_padding_reflect",
        "test_convtranspose3d_output_padding",
    ],
)
COREML_STATIC_INT8_TEST_FLOW = _create_coreml_flow(
    "coreml_static_int8",
    quantize=True,
    minimum_deployment_target=coremltools.target.iOS17,
)
