load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "vulkan_constant_materialization_tracker_test",
        srcs = ["test_vulkan_constant_materialization_tracker.cpp"],
        deps = [
            "//executorch/backends/native/runtime/vulkan:constant_materialization_tracker",
        ],
    )
