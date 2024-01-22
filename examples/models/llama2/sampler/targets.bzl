load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "sampler",
        exported_headers = ["sampler.h"],
        srcs = ["sampler.cpp"],
        visibility = [
            "//executorch/...",
        ],
    )

    runtime.cxx_test(
        name = "test_sampler",
        srcs = ["test/test_sampler.cpp"],
        deps = [
            ":sampler",
            "//caffe2:torch-cpp",
            "//executorch/runtime/platform:platform",
        ],
    )
