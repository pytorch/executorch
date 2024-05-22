load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "sgd_test",
        srcs = [
            "sgd_test.cpp",
        ],
        deps = [
            "//executorch/extension/training/optimizer:optimizer",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
