load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "scalar_type_util_test",
        srcs = ["ScalarTypeUtilTest.cpp"],
        deps = [
            "//executorch/core/kernel_types/util:scalar_type_util",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "tensor_util_test",
        srcs = ["TensorUtilTest.cpp"],
        deps = [
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/kernel_types/util:scalar_type_util",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "operator_impl_example_test",
        srcs = ["OperatorImplExampleTest.cpp"],
        deps = [
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/kernel_types/util:scalar_type_util",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "dim_order_util_test",
        srcs = ["DimOrderUtilsTest.cpp"],
        deps = [
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
    )
