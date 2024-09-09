load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "fast_hadamard_transform_test",
        srcs = ["fast_hadamard_transform_test.cpp"],
        headers = ["fast_hadamard_transform_special_unstrided_cpu.h"],
        deps = [
            "//executorch/extension/llm/custom_ops/spinquant:fast_hadamard_transform",
            "//executorch/extension/llm/custom_ops/spinquant/third-party/FFHT:dumb_fht",
        ],
    )
