load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    #
    # NOTE: tests for operators should live in kernels/portable/test, so that
    # they can be run against all implementations of a given operator. This
    # directory is only for testing cpu-specific helper libraries.
    #

    runtime.cxx_test(
        name = "scalar_utils_test",
        srcs = ["scalar_utils_test.cpp"],
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "vec_ops_test",
        srcs = ["vec_ops_test.cpp"],
        deps = ["//executorch/kernels/portable/cpu:vec_ops"],
    )
