# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Create flows for Arm Backends used to test operator and model suits

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.quantizer import get_symmetric_quantization_config
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.util._factory import create_quantizer
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.xnnpack.test.tester.tester import Quantize


def _create_arm_flow(
    name,
    compile_spec: ArmCompileSpec,
    symmetric_io_quantization: bool = False,
    per_channel_quantization: bool = True,
) -> TestFlow:

    def _create_arm_tester(*args, **kwargs) -> ArmTester:
        kwargs["compile_spec"] = compile_spec
        return ArmTester(*args, **kwargs)

    support_serialize = not isinstance(compile_spec, TosaCompileSpec)
    quantize = compile_spec.tosa_spec.support_integer()

    if quantize is True:

        def create_quantize_stage() -> Quantize:
            quantizer = create_quantizer(compile_spec)
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
        supports_serialize=support_serialize,
        quantize=quantize,
        quantize_stage_factory=(create_quantize_stage if quantize is True else False),
    )


ARM_TOSA_FP_FLOW = _create_arm_flow(
    "arm_tosa_fp",
    common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
)
ARM_TOSA_INT_FLOW = _create_arm_flow(
    "arm_tosa_int",
    common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
)
ARM_ETHOS_U55_FLOW = _create_arm_flow(
    "arm_ethos_u55",
    common.get_u55_compile_spec(),
)
ARM_ETHOS_U85_FLOW = _create_arm_flow(
    "arm_ethos_u85",
    common.get_u85_compile_spec(),
)
