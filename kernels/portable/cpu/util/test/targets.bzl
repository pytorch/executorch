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
        name = "broadcast_indexes_range_test",
        srcs = ["broadcast_indexes_range_test.cpp"],
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:broadcast_indexes_range",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
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

    # this test requires ET_USE_PYTORCH_HEADERS, which doesn't work in OSS Buck.
    if not runtime.is_oss:
        runtime.cxx_test(
            name = "vectorized_math_test",
            srcs = ["vectorized_math_test.cpp"],
            deps = [
                "//executorch/kernels/portable/cpu/util:vectorized_math",
                "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
            ],
        )
