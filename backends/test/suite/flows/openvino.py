# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from executorch.backends.openvino.test.tester import (
    OpenVINOTester,
    Quantize as OpenVINOQuantize,
)
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow


def _create_openvino_flow_base(
    name: str,
    quantize_stage_factory: Optional[Callable[..., Quantize]] = None,
) -> TestFlow:
    return TestFlow(
        name,
        backend="openvino",
        tester_factory=OpenVINOTester,
        quantize=quantize_stage_factory is not None,
        quantize_stage_factory=quantize_stage_factory,
        skip_patterns=[
            "test_avgpool1d_combinations",
            "test_avgpool3d_combinations",
            "test_conv1d_padding_modes",
            "test_conv2d_padding_modes",
            "test_embedding_bag_include_last_offset",
            "test_embedding_bag_modes",
            "test_threshold_f32_all_params",
            "test_transpose_identity",
            "test_convnext_small",
            "test_shufflenet_v2_x1_0",
            "test_swin_v2_t",
        ],
    )


def _create_openvino_flow() -> TestFlow:
    return _create_openvino_flow_base("openvino")


def _create_openvino_int8_flow() -> TestFlow:
    """
    INT8 quantization flow for OpenVINO.
    Uses post-training quantization with calibration.
    """

    def create_quantize_stage() -> Quantize:
        return OpenVINOQuantize(calibrate=True)

    return _create_openvino_flow_base("openvino_int8", create_quantize_stage)


OPENVINO_TEST_FLOW = _create_openvino_flow()
OPENVINO_INT8_TEST_FLOW = _create_openvino_int8_flow()
