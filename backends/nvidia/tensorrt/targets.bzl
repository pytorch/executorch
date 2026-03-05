load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "backend",
        srcs = [
            "backend.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
            "//executorch/exir/backend:backend_details",
        ],
    )

    runtime.python_library(
        name = "tensorrt",
        srcs = [
            "__init__.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":backend",
        ],
    )

    runtime.python_library(
        name = "converter_registry",
        srcs = [
            "converter_registry.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "converter_utils",
        srcs = [
            "converter_utils.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
            "//deeplearning/trt/python:py_tensorrt",
        ],
    )
