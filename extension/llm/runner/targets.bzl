load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "stats",
        headers = "stats.h",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )
