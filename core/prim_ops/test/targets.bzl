load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "register_prim_ops_test",
        srcs = [
            "RegisterPrimOpsTest.cpp",
        ],
        deps = [
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/prim_ops:prim_ops_registry",
            "//executorch/core:operator_registry",
            "//executorch/kernels:kernel_runtime_context",
        ],
    )
