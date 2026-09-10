load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "owned_bytes_test",
        srcs = ["test_owned_bytes.cpp"],
        deps = [
            "//executorch/backends/native/runtime/deserialize:owned_bytes",
        ],
    )

    runtime.cxx_test(
        name = "safetensors_reader_test",
        srcs = ["test_safetensors_reader.cpp"],
        deps = [
            "//executorch/backends/native/runtime/deserialize:safetensors_reader",
        ],
    )

    runtime.cxx_test(
        name = "zip_reader_test",
        srcs = ["test_zip_reader.cpp"],
        deps = [
            "//executorch/backends/native/runtime/deserialize:zip_reader",
            "fbsource//third-party/libzip:zip",
        ],
    )
