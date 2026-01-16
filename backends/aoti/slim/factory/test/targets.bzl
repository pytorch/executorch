load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define test targets for SlimTensor factory module."""

    runtime.cxx_test(
        name = "test_empty",
        srcs = [
            "test_empty.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/factory:empty",
        ],
    )
