load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define test targets for SlimTensor util module."""

    runtime.cxx_test(
        name = "test_size_util",
        srcs = [
            "test_size_util.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/util:size_util",
        ],
    )
