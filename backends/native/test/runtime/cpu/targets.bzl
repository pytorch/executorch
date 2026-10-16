load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "cpu_engine_test",
        srcs = ["test_cpu_engine.cpp"],
        deps = [
            "//executorch/backends/cpu:cpu_reference",
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/api:api",
            "//executorch/backends/native/runtime/cpu:cpu_engine",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/deserialize:package_test_data",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )
