load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "bfloat16_test",
        srcs = ["bfloat16_test.cpp"],
        deps = [
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )

    runtime.cxx_test(
        name = "optional_test",
        srcs = ["optional_test.cpp"],
        deps = [
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )

    runtime.cxx_test(
        name = "tensor_test",
        srcs = ["tensor_test.cpp"],
        deps = [
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )

    runtime.cxx_test(
        name = "half_test",
        srcs = ["half_test.cpp"],
        deps = [
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )

    runtime.cxx_test(
        name = "scalar_test",
        srcs = ["scalar_test.cpp"],
        deps = [
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )

    runtime.cxx_test(
        name = "tensor_impl_test",
        srcs = ["tensor_impl_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )
