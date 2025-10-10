load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "temp_file",
        srcs = [],
        exported_headers = ["temp_file.h"],
        visibility = [
            "//executorch/devtools/etdump/tests/...",
            "//executorch/extension/data_loader/test/...",
            "//executorch/extension/llm/runner/test/...",
            "//executorch/extension/testing_util/test/...",
            "//executorch/extension/fb/ptez/decompression_methods/test/...",
            "//executorch/extension/fb/ptez/test/...",
            "//executorch/runtime/executor/test/...",
            "//executorch/backends/xnnpack/test/...",
        ],
    )
