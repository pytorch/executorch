load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor core module."""

    # Header-only library for Storage
    runtime.cxx_library(
        name = "storage",
        exported_headers = [
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
        ] + (
            [
                "//executorch/backends/aoti/slim/c10/cuda:exception",
                "//executorch/backends/aoti/slim/cuda:guard",
                "//executorch/extension/cuda:caller_stream",
                "//executorch/extension/cuda:cuda_allocator",
            ] if is_fbcode() or runtime.is_oss else select({
                "DEFAULT": [],
                "ovr_config//cuda:has_cuda": [
                    "//executorch/backends/aoti/slim/c10/cuda:exception",
                    "//executorch/backends/aoti/slim/cuda:guard",
                    "//executorch/extension/cuda:caller_stream",
                    "//executorch/extension/cuda:cuda_allocator",
                ],
            })
        ),
    )

    runtime.cxx_library(
        name = "slimtensor",
        exported_headers = [
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
        ],
    )
