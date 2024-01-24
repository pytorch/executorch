load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "sampler",
        exported_headers = [
            "sampler.h",
        ],
        srcs = [
            "sampler.cpp",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//executorch/...",
        ],
    )
