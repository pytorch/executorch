load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "native_executor",
        srcs = [
            "native_executor.cpp",
        ],
        exported_headers = [
            "native_executor.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/backends/native/runtime/engine:engine",
            "//executorch/backends/native/runtime/graph:scalar_type",
            "//executorch/extension/llm/batching:batching",
            "//executorch/runtime/core:core",
        ],
        deps = [
            "//executorch/backends/native/runtime:offgraph_cache",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/extension/llm/batching:session_table",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/extension/llm/runner:constants",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/platform:platform",
        ],
    )
