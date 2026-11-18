load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/backends/vulkan:targets.bzl",
    "get_platforms",
)

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.cxx_test(
        name = "q8ta_conv2d_stream_plan_test",
        srcs = ["q8ta_conv2d_stream_plan_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
    )

    runtime.cxx_test(
        name = "q8ta_conv2d_route_test",
        srcs = ["q8ta_conv2d_route_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
    )
