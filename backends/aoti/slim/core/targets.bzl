load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor core module."""

    # Header-only library for Storage
    runtime.cxx_library(
        name = "storage",
        headers = [
            "Storage.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/core:scalar_type",
            "//executorch/backends/aoti/slim/util:shared_ptr",
            "//executorch/runtime/platform:platform",
        ],
    )
