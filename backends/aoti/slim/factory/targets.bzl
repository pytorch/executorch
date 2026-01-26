load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor factory module."""

    # Header-only library for empty tensor factory functions
    runtime.cxx_library(
        name = "empty",
        headers = [
            "Empty.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/backends/aoti/slim/util:size_util",
        ],
    )

    runtime.cxx_library(
        name = "from_blob",
        headers = [
            "FromBlob.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/backends/aoti/slim/util:size_util",
        ],
    )
