import argparse
import unittest

import nncf.torch  # type: ignore[import-untyped,import-not-found]


class OpenvinoTestSuite(unittest.TestSuite):

    test_params: dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addTest(self, test):
        # Set test parameters if this is an instance of TestOpenvino
        from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
            BaseOpenvinoOpTest,
        )

        if isinstance(test, BaseOpenvinoOpTest):
            if "device" in self.test_params:
                test.device = self.test_params["device"]
        # Call the original addTest method to actually add the test to the suite
        super().addTest(test)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--device",
        help="OpenVINO device to execute the model on",
        type=str,
        default="CPU",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        help="Pattern to match test files. Provide complete file name to run individual tests",
        type=str,
        default="test_*.py",
    )
    parser.add_argument(
        "-t",
        "--test_type",
        help="Specify the type of tests ('ops', 'models' or 'quantizer')",
        type=str,
        default="ops",
        choices={"ops", "models", "quantizer"},
    )

    args, ns_args = parser.parse_known_args(namespace=unittest)
    test_params: dict[str, str] = {}
    test_params["device"] = args.device
    test_params["pattern"] = args.pattern
    test_params["test_type"] = args.test_type
    return test_params


if __name__ == "__main__":
    loader = unittest.TestLoader()
    # Replace the default test suite with a custom test suite to be able to
    # pass test parameter to the test cases
    loader.suiteClass = OpenvinoTestSuite
    test_params = parse_arguments()
    loader.suiteClass.test_params = test_params
    # Discover all existing op tests in "ops" folder
    suite = loader.discover(test_params["test_type"], pattern=test_params["pattern"])
    # Start running tests
    # with nncf.torch.disable_patching():
    result = unittest.TextTestRunner().run(suite)
    if result.wasSuccessful():
        print("OpenVINO backend tests completed successfully")
    else:
        print("OpenVINO backend tests completed with failures")
