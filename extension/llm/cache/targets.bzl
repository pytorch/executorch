load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Mirrors the extension_llm_cache CMake target: the neutral core
    # (cache.h, sequence_cache.h, cell_cache) and the ExecuTorch rendezvous
    # registry build as one unit. See README.md for the split they are meant
    # to have eventually.
    runtime.cxx_library(
        name = "kv_cache",
        srcs = [
            "cache_registry.cpp",
            "cell_cache.cpp",
        ],
        exported_headers = [
            "cache.h",
            "cache_registry.h",
            "cell_cache.h",
            "sequence_cache.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )
