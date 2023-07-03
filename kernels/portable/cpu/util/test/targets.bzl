load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "broadcast_test",
        srcs = ["broadcast_test.cpp"],
        deps = [
            "//executorch/core/kernel_types:kernel_types",
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/core/values:executor_values",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    )

    runtime.cxx_test(
        name = "reduce_test",
        srcs = ["reduce_test.cpp"],
        deps = [
            "//executorch/core/kernel_types:kernel_types",
            "//executorch/core/kernel_types/testing:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    )
