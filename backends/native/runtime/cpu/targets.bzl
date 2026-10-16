load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "cpu_engine",
        srcs = ["CPUEngine.cpp", "CPUEngineExecutable.cpp"],
        headers = ["CPUEngineExecutable.h"],
        exported_headers = ["CPUEngine.h"],
        exported_deps = [
            "//executorch/backends/cpu:cpu_runtime",
            "//executorch/backends/native/runtime/engine:engine",
        ],
        deps = [
            "//executorch/backends/cpu:cpu_reference",
            "//executorch/backends/cpu:cpu_xnnpack",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/graph:graph_utils",
            "//executorch/configurations:optimized_native_cpu_ops",
            "//executorch/extension/memory_allocator:malloc_memory_allocator",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        visibility = ["PUBLIC"],
    )
