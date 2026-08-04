load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")

def define_common_targets():
    """Define test targets for SlimTensor util module."""

    if not is_fbcode():
        return

    runtime.cxx_test(
        name = "test_size_util",
        srcs = [
            "test_size_util.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/util:size_util",
        ],
    )
