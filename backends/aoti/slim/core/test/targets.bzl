load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define test targets for SlimTensor core module."""

    runtime.cxx_test(
        name = "test_storage",
        srcs = [
            "test_storage.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )
