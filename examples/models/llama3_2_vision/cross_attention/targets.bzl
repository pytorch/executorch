load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "cross_attention_mask",
        srcs = ["cross_attention_mask.cpp"],
        exported_headers = ["cross_attention_mask.h"],
        exported_deps = [
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "cross_attention_mask_test",
        srcs = ["cross_attention_mask_test.cpp"],
        deps = [":cross_attention_mask"],
    )
