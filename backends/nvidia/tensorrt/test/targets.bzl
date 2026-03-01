load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_test(
        name = "test_converter_registry",
        srcs = [
            "test_converter_registry.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/nvidia/tensorrt:converter_registry",
            "//executorch/backends/nvidia/tensorrt:converter_utils",
            "//executorch/backends/nvidia/tensorrt/converters:converters",
        ],
    )

    runtime.python_test(
        name = "test_operator_support",
        srcs = [
            "test_operator_support.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/nvidia/tensorrt/partitioner:partitioner",
            "//executorch/exir:lib",
        ],
    )
