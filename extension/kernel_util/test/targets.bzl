load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "make_boxed_from_unboxed_functor_test",
        srcs = [
            "make_boxed_from_unboxed_functor_test.cpp",
        ],
        deps = [
            "//executorch/extension/kernel_util:kernel_util",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
