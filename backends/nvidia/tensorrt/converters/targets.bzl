load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "converters",
        srcs = [
            "__init__.py",
            "add.py",
            "div.py",
            "mm.py",
            "mul.py",
            "relu.py",
            "sub.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/backends/nvidia/tensorrt:converter_registry",
            "//executorch/backends/nvidia/tensorrt:converter_utils",
        ],
    )
