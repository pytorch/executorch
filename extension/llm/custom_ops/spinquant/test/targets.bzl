load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "fast_hadamard_transform_test_impl",
        srcs = ["fast_hadamard_transform_test_impl.cpp"],
        exported_headers = [
            "fast_hadamard_transform_special_unstrided_cpu.h",
            "fast_hadamard_transform_test_impl.h",
        ],
        exported_deps = [
            "//executorch/extension/llm/custom_ops/spinquant:fast_hadamard_transform",
        ],
        deps = [
            "//executorch/extension/llm/custom_ops/spinquant/third-party/FFHT:dumb_fht",
        ],
    )

    runtime.cxx_test(
        name = "fast_hadamard_transform_test",
        srcs = ["fast_hadamard_transform_test.cpp"],
        deps = [
            ":fast_hadamard_transform_test_impl",
            "//executorch/extension/llm/custom_ops/spinquant:fast_hadamard_transform",
        ],
    )

    runtime.cxx_test(
        name = "op_fast_hadamard_transform_test",
        srcs = ["op_fast_hadamard_transform_test.cpp"],
        deps = [
            ":fast_hadamard_transform_test_impl",
            "//executorch/extension/llm/custom_ops:custom_ops",
            "//executorch/kernels/test:test_util",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
