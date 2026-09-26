load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "program_test",
        srcs = ["test_program_deserialize.cpp"],
        deps = [
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime:runtime",
        ],
    )
