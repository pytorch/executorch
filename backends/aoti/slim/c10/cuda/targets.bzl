load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")
load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor CUDA exception handling module."""


    # Outside fbcode the CUDA headers resolve only on CUDA platforms.
    cuda_kwargs = {} if is_fbcode() or runtime.is_oss else {
        "platforms": [CXX],
        "compatible_with": ["ovr_config//cuda:has_cuda"],
    }

    runtime.cxx_library(
        name = "exception",
        exported_headers = [
            "Exception.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/macros:macros",
            "//executorch/extension/cuda:runtime_api",
            "//executorch/runtime/platform:platform",
        ],
        **cuda_kwargs
    )
