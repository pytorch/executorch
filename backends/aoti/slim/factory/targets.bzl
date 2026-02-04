load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor factory module."""

    # Header-only library for empty tensor factory functions
    runtime.cxx_library(
        name = "empty",
        headers = [
            "empty.h",
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
            "from_blob.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/backends/aoti/slim/util:size_util",
        ],
    )

    runtime.cxx_library(
        name = "from_etensor",
        headers = [
            "from_etensor.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/factory:empty",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/runtime/core/portable_type:portable_type",
        ],
    )
