import unittest
import argparse

class OpenvinoTestSuite(unittest.TestSuite):

    test_params = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addTest(self, test):
        # Set test parameters if this is an instance of TestOpenvino
        from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
        if isinstance(test, BaseOpenvinoOpTest):
            if "device" in self.test_params:
                test.device = self.test_params["device"]
            if "build_folder" in self.test_params:
                test.build_folder = self.test_params["build_folder"]
        # Call the original addTest method to actually add the test to the suite
        super().addTest(test)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="OpenVINO device to execute the model on",
        type=str,
        default="CPU",
    )

    args, ns_args = parser.parse_known_args(namespace=unittest)
    test_params = {}
    test_params["device"] = args.device
    test_params["build_folder"] = args.build_folder
    return test_params

if __name__ == "__main__":
    loader = unittest.TestLoader()
    # Replace the default test suite with a custom test suite to be able to
    # pass test parameter to the test cases
    loader.suiteClass = OpenvinoTestSuite
    loader.suiteClass.test_params = parse_arguments()
    # Discover all existing op tests in "ops" folder
    suite = loader.discover("ops", pattern='test_*.py')
    # Start running tests
    unittest.TextTestRunner().run(suite)
