load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor util module."""

    # Header-only library for SharedPtr
    runtime.cxx_library(
        name = "shared_ptr",
        headers = [
            "SharedPtr.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/platform:platform",
        ],
    )

    # Header-only library for ArrayRefUtil
    runtime.cxx_library(
        name = "array_ref_util",
        headers = [
            "ArrayRefUtil.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    # Header-only library for SizeUtil
    runtime.cxx_library(
        name = "size_util",
        headers = [
            "SizeUtil.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":array_ref_util",
            "//executorch/runtime/platform:platform",
        ],
    )
