load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    # AOT export coverage only (lowers via VulkanPartitioner, asserts a VulkanBackend delegate); no GPU runtime.
    python_unittest(
        name = "test_add",
        srcs = [
            "ops/add/test_add.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
            "//executorch/backends/vulkan:vulkan_preprocess",
            "//executorch/exir:lib",
        ],
    )

    runtime.python_library(
        name = "tester",
        srcs = ["tester.py"],
        deps = [
            "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
            "//executorch/backends/vulkan:vulkan_preprocess",
        ],
    )
