load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor CUDA exception handling module."""

    runtime.cxx_library(
        name = "exception",
        exported_headers = [
            "Exception.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/macros:macros",
            "//executorch/runtime/platform:platform",
        ],
    )
