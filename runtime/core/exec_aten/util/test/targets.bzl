load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "scalar_type_util_test",
        srcs = ["scalar_type_util_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "operator_impl_example_test",
        srcs = ["operator_impl_example_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "dim_order_util_test",
        srcs = ["dim_order_util_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_test(
            name = "tensor_util_test" + aten_suffix,
            srcs = ["tensor_util_test.cpp"],
            deps = [
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
                "//executorch/runtime/core/exec_aten/util:scalar_type_util",
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
        )
