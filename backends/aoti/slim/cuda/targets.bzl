load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor CUDA guard module."""

    runtime.cxx_library(
        name = "guard",
        srcs = [
            "guard.cpp",
        ],
        headers = [
            "guard.h",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/platform:platform",
        ],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/cuda:exception",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )
