from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.test.suite.flow import TestFlow

def _create_arm_tester_tosa_fp(*args, **kwargs) -> ArmTester:
    kwargs["compile_spec"] = common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP")
    
    return ArmTester(
        *args,
        **kwargs,
    )

def _create_tosa_flow() -> TestFlow:
    return TestFlow(
        "arm_tosa",
        backend="arm",
        tester_factory=_create_arm_tester_tosa_fp,
        supports_serialize=False,
    )

ARM_TOSA_FLOW = _create_tosa_flow()