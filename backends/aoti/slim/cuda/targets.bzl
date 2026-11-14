load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")
load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor CUDA guard module."""


    # Outside fbcode the CUDA headers resolve only on CUDA platforms.
    cuda_kwargs = {} if is_fbcode() or runtime.is_oss else {
        "platforms": [CXX],
        "compatible_with": ["ovr_config//cuda:has_cuda"],
    }

    runtime.cxx_library(
        name = "guard",
        srcs = [
            "guard.cpp",
        ],
        exported_headers = [
            "guard.h",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/platform:platform",
        ],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/cuda:exception",
            "//executorch/extension/cuda:caller_stream",
            "//executorch/extension/cuda:runtime_api",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
        **cuda_kwargs
    )
