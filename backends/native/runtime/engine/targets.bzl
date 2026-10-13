load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # The runtime <-> compute-backend boundary: EngineHost owns process-wide
    # device state, EngineContext owns one loaded program, and EngineExecutable
    # runs one lowered Method.
    runtime.cxx_library(
        name = "engine",
        srcs = ["Engine.cpp"],
        exported_headers = [
            "Engine.h",
        ],
        deps = [
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime:validation",
            "//executorch/backends/native/runtime/deserialize:package",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "ggml_engine_cpu",
        srcs = [
            "ggml/GgmlEngine.cpp",
            "ggml/GgmlExecutor.cpp",
        ],
        headers = ["ggml/GgmlExecutor.h"],
        exported_headers = ["ggml/GgmlEngine.h"],
        exported_deps = [":engine"],
        deps = [
            "//executorch/backends/native/runtime:method",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/graph:argument",
            "//executorch/backends/native/runtime/graph:graph",
            "//executorch/backends/native/runtime/graph:ids",
            "//executorch/backends/native/runtime/graph:node",
            "//executorch/backends/native/runtime/graph:scalar_type",
            "//executorch/backends/native/runtime/graph:tensor_meta",
            "//executorch/backends/native/runtime/graph:value",
            "fbsource//third-party/llama.cpp:ggml_cpu",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "ggml_engine_cuda",
        srcs = [
            "ggml/GgmlEngine.cpp",
            "ggml/GgmlExecutor.cpp",
        ],
        headers = ["ggml/GgmlExecutor.h"],
        exported_headers = ["ggml/GgmlEngine.h"],
        exported_deps = [":engine"],
        preprocessor_flags = ["-DPTN_GGML_USE_CUDA=1"],
        deps = [
            "//executorch/backends/native/runtime:method",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/graph:argument",
            "//executorch/backends/native/runtime/graph:graph",
            "//executorch/backends/native/runtime/graph:ids",
            "//executorch/backends/native/runtime/graph:node",
            "//executorch/backends/native/runtime/graph:scalar_type",
            "//executorch/backends/native/runtime/graph:tensor_meta",
            "//executorch/backends/native/runtime/graph:value",
            "fbsource//third-party/llama.cpp:ggml_cuda",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
