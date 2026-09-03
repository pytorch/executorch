load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Minimal JSON reader for the package's index metadata (safetensors header and
    # alias map). Integers only; see JsonParser.h.
    runtime.cxx_library(
        name = "json_parser",
        srcs = ["JsonParser.cpp"],
        exported_headers = ["JsonParser.h"],
        visibility = ["//executorch/backends/native/..."],
    )
