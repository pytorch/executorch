load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Borrowed byte-range view shared by the package readers (a std::span alias,
    # named so the borrow contract has somewhere to live).
    runtime.cxx_library(
        name = "byte_span",
        srcs = [],
        exported_headers = ["ByteSpan.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Minimal JSON reader for the package's index metadata (safetensors header and
    # alias map). Integers only; see JsonParser.h.
    runtime.cxx_library(
        name = "json_parser",
        srcs = ["JsonParser.cpp"],
        exported_headers = ["JsonParser.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Read-only reader for stored (uncompressed) zip archives, which is what a .ptn
    # package is.
    runtime.cxx_library(
        name = "zip_reader",
        srcs = ["ZipReader.cpp"],
        exported_headers = ["ZipReader.h"],
        exported_deps = [":byte_span"],
        visibility = ["//executorch/backends/native/..."],
    )

    # safetensors index reader.
    runtime.cxx_library(
        name = "safetensors_reader",
        srcs = ["SafeTensorsReader.cpp"],
        exported_headers = ["SafeTensorsReader.h"],
        exported_deps = [
            ":byte_span",
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        deps = [":json_parser"],
        visibility = ["//executorch/backends/native/..."],
    )
