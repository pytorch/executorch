# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.xnnpack.test.tester.tester import Quantize


def _create_tosa_flow(
    name,
    compile_spec,
    quantize: bool = False,
    symmetric_io_quantization: bool = False,
    per_channel_quantization: bool = True,
) -> TestFlow:

    def _create_arm_tester(*args, **kwargs) -> ArmTester:
        kwargs["compile_spec"] = compile_spec

        return ArmTester(
            *args,
            **kwargs,
        )

    # Create and configure quantizer to use in the flow
    def create_quantize_stage() -> Quantize:
        quantizer = TOSAQuantizer(compile_spec)
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=per_channel_quantization
        )
        if symmetric_io_quantization:
            quantizer.set_io(quantization_config)
        return Quantize(quantizer, quantization_config)

    return TestFlow(
        name,
        backend="arm",
        tester_factory=_create_arm_tester,
        supports_serialize=False,
        quantize=quantize,
        quantize_stage_factory=create_quantize_stage if quantize else None,
    )


ARM_TOSA_FP_FLOW = _create_tosa_flow(
    "arm_tosa_fp", common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP")
)
ARM_TOSA_INT_FLOW = _create_tosa_flow(
    "arm_tosa_int",
    common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
    quantize=True,
)
