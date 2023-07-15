load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "register_prim_ops_test",
        srcs = [
            "register_prim_ops_test.cpp",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/kernels/prim_ops:prim_ops_registry",
            "//executorch/runtime/kernel:operator_registry",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )
