load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "safetensors_reader_test",
        srcs = ["test_safetensors_reader.cpp"],
        deps = [
            "//executorch/backends/native/runtime/deserialize:safetensors_reader",
        ],
    )
