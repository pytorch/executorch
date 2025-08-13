from executorch.backends.qualcomm.tests.tester import QualcommTester
from executorch.backends.test.suite.flow import TestFlow


def _create_qualcomm_flow(
    name: str,
    quantize: bool = False,
) -> TestFlow:
    return TestFlow(
        name,
        backend="qualcomm",
        tester_factory=QualcommTester,
        quantize=quantize,
    )


QUALCOMM_TEST_FLOW = _create_qualcomm_flow("qualcomm")
