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
            "//executorch/backends/native/runtime/deserialize:package",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
