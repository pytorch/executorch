load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "EvalueTest",
        srcs = ["EvalueTest.cpp"],
        deps = [
            "//executorch/core/kernel_types:kernel_types",
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/values:executor_values",
        ],
    )

    runtime.cxx_test(
        name = "EvalueTest_aten",
        srcs = ["EvalueTest.cpp"],
        deps = [
            "//executorch/core/kernel_types:kernel_types_aten",
            "//executorch/core/kernel_types/testing:tensor_util_aten",
            "//executorch/core/values:executor_values_aten",
        ],
    )
