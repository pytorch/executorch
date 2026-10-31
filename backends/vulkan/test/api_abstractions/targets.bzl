load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/vulkan:targets.bzl", "get_platforms")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.cxx_test(
        name = "dispatch_grid_test",
        srcs = ["dispatch_grid_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_compute_api",
            "//third-party/googletest:gtest_main",
        ],
    )
