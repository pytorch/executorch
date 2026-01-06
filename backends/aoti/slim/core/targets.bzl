load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor core module."""

    # Header-only library for Storage
    runtime.cxx_library(
        name = "storage",
        headers = [
            "storage.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/core:scalar_type",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/backends/aoti/slim/util:shared_ptr",
            "//executorch/backends/aoti/slim/util:size_util",
            "//executorch/runtime/platform:platform",
            "//executorch/backends/aoti/slim/c10/cuda:exception",
            "//executorch/backends/aoti/slim/cuda:guard",
        ],
    )

    runtime.cxx_library(
        name = "slimtensor",
        headers = [
            "slim_tensor.h",
            "slim_tensor_view_incl.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":storage",
            "//executorch/backends/aoti/slim/c10/core:contiguity",
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/core:scalar_type",
            "//executorch/backends/aoti/slim/c10/core:sizes_and_strides",
            "//executorch/backends/aoti/slim/c10/core:wrap_dim_minimal",
            "//executorch/backends/aoti/slim/util:array_ref_util",
            "//executorch/backends/aoti/slim/util:size_util",
            "//executorch/runtime/platform:platform",
            "//executorch/backends/aoti/slim/c10/cuda:exception",
            "//executorch/backends/aoti/slim/cuda:guard",
        ],
    )
