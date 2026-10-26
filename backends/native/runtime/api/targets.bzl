load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Keep the standalone API independent of ExecuTorch runtime and extension
    # libraries.
    runtime.cxx_library(
        name = "api",
        srcs = ["Model.cpp"],
        exported_headers = [
            "MethodInfo.h",
            "Model.h",
            "Session.h",
            "Tensor.h",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime:method_meta",
            "//executorch/backends/native/runtime:tensor_info",
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        deps = [
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime:validation",
            "//executorch/backends/native/runtime/deserialize:checked_math",
            "//executorch/backends/native/runtime/deserialize:limits",
            "//executorch/backends/native/runtime/deserialize:owned_bytes",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/engine:engine",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "api_types_test",
        srcs = ["test/ApiTypesTest.cpp"],
        deps = [
            ":api",
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/deserialize:package_test_data",
            "//executorch/backends/native/runtime/engine:engine",
        ],
    )
