load("//caffe2/test/fb:defs.bzl", "define_tests")
load("@bazel_skylib//lib:paths.bzl", "paths")

def define_arm_tests():
    # TODO Add more tests
    test_files = native.glob(["passes/test_*.py"])

    TESTS = {}

    for test_file in test_files:
        test_file_name = paths.basename(test_file)
        test_name = test_file_name.replace("test_", "").replace(".py", "")
        TESTS[test_name] = [test_file]

    define_tests(
        pytest = True,
        tests = TESTS,
        pytest_config = "pytest.ini",
        resources = ["conftest.py"],
        preload_deps = [
            "//executorch/kernels/quantized:custom_ops_generated_lib",
        ],
        deps = [
            ":arm_tester",
            ":conftest",
            "//executorch/exir:lib",
            "fbsource//third-party/pypi/pytest:pytest",
            "fbsource//third-party/pypi/parameterized:parameterized",
        ],
    )
