load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "checked_math",
        srcs = [],
        exported_headers = ["CheckedMath.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "deserialize_error",
        srcs = [],
        exported_headers = ["DeserializeError.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "limits",
        srcs = [],
        exported_headers = ["Limits.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Borrowed byte-range view shared by the package readers (a std::span alias,
    # named so the borrow contract has somewhere to live).
    runtime.cxx_library(
        name = "byte_span",
        srcs = [],
        exported_headers = ["ByteSpan.h"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Owning byte buffer behind a package: heap read or read-only mmap.
    runtime.cxx_library(
        name = "owned_bytes",
        srcs = ["OwnedBytes.cpp"],
        exported_headers = ["OwnedBytes.h"],
        exported_deps = [":byte_span"],
        deps = [":deserialize_error"],
        visibility = ["//executorch/backends/native/..."],
    )

    # JSON representation used by package metadata readers.
    runtime.cxx_library(
        name = "json",
        srcs = [],
        exported_headers = ["Json.h"],
        exported_deps = [
            ":deserialize_error",
            ":limits",
        ],
        exported_external_deps = ["nlohmann_json"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Read-only reader for stored (uncompressed) zip archives, which is what a .ptn
    # package is.
    runtime.cxx_library(
        name = "zip_reader",
        srcs = ["ZipReader.cpp"],
        exported_headers = ["ZipReader.h"],
        exported_deps = [":byte_span"],
        deps = [
            "fbsource//third-party/libzip:zip",
            ":deserialize_error",
            ":limits",
        ],
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
        deps = [
            ":checked_math",
            ":deserialize_error",
            ":json",
            ":limits",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
    # The .ptn package: program flatbuffer plus its constants.
    runtime.cxx_library(
        name = "package",
        srcs = ["Package.cpp"],
        exported_headers = ["Package.h"],
        exported_deps = [
            ":byte_span",
            ":owned_bytes",
            ":safetensors_reader",
            ":zip_reader",
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        deps = [
            ":deserialize_error",
            ":json",
            ":limits",
        ],
        visibility = ["PUBLIC"],
    )
    runtime.cxx_test(
        name = "reader_bounds_test",
        srcs = ["test/ReaderBoundsTest.cpp"],
        deps = [
            ":deserialize_error",
            ":json",
            ":limits",
            ":safetensors_reader",
            ":zip_reader",
        ],
    )

    runtime.cxx_library(
        name = "package_test_data",
        exported_headers = ["test/PackageTestData.h"],
        visibility = ["//executorch/backends/native/..."],
    )
