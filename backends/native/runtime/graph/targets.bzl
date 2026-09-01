load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "string_format",
        exported_headers = [
            "StringFormat.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
