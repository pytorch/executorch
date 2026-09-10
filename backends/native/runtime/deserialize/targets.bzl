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

    # JSON representation used by package metadata readers.
    runtime.cxx_library(
        name = "json",
        srcs = [],
        exported_headers = ["Json.h"],
        exported_external_deps = ["nlohmann_json"],
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
        deps = [":json"],
        visibility = ["//executorch/backends/native/..."],
    )
