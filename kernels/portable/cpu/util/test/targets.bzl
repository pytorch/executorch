load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "broadcast_test",
        srcs = ["broadcast_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core:evalue",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    )

    runtime.cxx_test(
        name = "reduce_test",
        srcs = ["reduce_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    )
