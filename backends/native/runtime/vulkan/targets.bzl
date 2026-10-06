load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "constant_materialization_tracker",
        srcs = ["VulkanConstantMaterializationTracker.cpp"],
        exported_headers = [
            "VulkanConstantMaterializationTracker.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
