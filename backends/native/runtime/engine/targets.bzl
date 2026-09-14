load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # The runtime <-> compute-backend boundary: the abstract EngineContext
    # (process-wide device state) and EngineExecutable (one lowered, ready-to-run
    # Method). Pure std; each backend implements both in its own package.
    runtime.cxx_library(
        name = "engine",
        srcs = ["Engine.cpp"],
        exported_headers = [
            "Engine.h",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime:method",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
