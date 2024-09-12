load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "fast_hadamard_transform",
        exported_headers = [
            "fast_hadamard_transform.h",
            "fast_hadamard_transform_special.h",
        ],
        srcs = [
            "fast_hadamard_transform.cpp",
        ],
        exported_deps = [
            "//executorch/extension/llm/custom_ops/spinquant/third-party/FFHT:fht",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
    )
