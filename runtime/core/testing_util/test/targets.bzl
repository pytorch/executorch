
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "test_error_matchers",
        srcs = [
            "test_error_matchers.cpp",
        ],
        visibility = ["//executorch/..."],
        deps = [
            "//executorch/runtime/core/testing_util:error_matchers",
        ],
    )
