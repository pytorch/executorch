load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "partitioner",
        srcs = [
            "__init__.py",
            "operator_support.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/nvidia/tensorrt:backend",
            "//executorch/exir/backend:partitioner",
        ],
    )
