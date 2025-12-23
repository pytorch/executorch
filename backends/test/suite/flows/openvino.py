# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable

from executorch.backends.openvino.test.tester import (
    Quantize as OpenVINOQuantize,
    OpenVINOTester,
)
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_openvino_flow_base(
    name: str,
    quantize: bool = False,
    compile_specs: dict | None = None,
) -> TestFlow:
    logger.info("Creating OPENVINO FLOW test flow")
    logger.info(f"NAME: {name}")
    return TestFlow(
        name,
        backend="openvino",
        tester_factory=OpenVINOTester,
        quantize=quantize,
    )


def _create_openvino_flow() -> TestFlow:
    logger.info("Creating OpenVINO FP32 test flow")
    return _create_openvino_flow_base("openvino")


def _create_openvino_int8_flow() -> TestFlow:
    """
    INT8 quantization flow for OpenVINO.
    Uses post-training quantization with calibration.
    """
    print("In OpenVINO INT8 FLOW")

    def create_quantize_stage() -> Quantize:
        return OpenVINOQuantize(
            calibrate=True,
        )

    return _create_openvino_flow_base("openvino_int8", create_quantize_stage)


OPENVINO_TEST_FLOW = _create_openvino_flow()
OPENVINO_INT8_TEST_FLOW = _create_openvino_int8_flow()
