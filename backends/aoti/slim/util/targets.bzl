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
