load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/vulkan:targets.bzl", "get_platforms")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.cxx_library(
        name = "test_utils",
        exported_headers = ["VulkanTest.h"],
        platforms = get_platforms(),
        exported_deps = [
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//third-party/googletest:gtest_main",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "profiling_api_test",
        srcs = ["profiling_api_test.cpp"],
        platforms = get_platforms(),
        deps = [
            ":test_utils",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_test(
        name = "dispatch_grid_test",
        srcs = ["dispatch_grid_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_compute_api",
            "//third-party/googletest:gtest_main",
        ],
    )
