load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "format",
        exported_headers = [
            "Format.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
