load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():

    for aten in get_aten_mode_options():
        aten_suffix = "_aten" if aten else ""

        # Interface for IOManager. No concrete impl from this dep.
        runtime.cxx_library(
            name = "io_manager" + aten_suffix,
            exported_headers = [
                "io_manager.h",
            ],
            exported_deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
            ],
            visibility = ["PUBLIC"],
        )

        runtime.cxx_library(
            name = "hf_static_io_manager" + aten_suffix,
            exported_headers = ["hf_static_io_manager.h"],
            srcs = ["hf_static_io_manager.cpp"],
            exported_deps = [
                ":io_manager" + aten_suffix,
                "//executorch/extension/llm/runner:constants",
            ],
            visibility = ["PUBLIC"],
        )

        # IOManager for the "kvcache_io" contract: KV cache as method I/O.
        runtime.cxx_library(
            name = "kv_cache_io_manager" + aten_suffix,
            exported_headers = [
                "kv_cache_io_manager.h",
            ],
            srcs = [
                "kv_cache_io_manager.cpp",
            ],
            exported_deps = [
                ":io_manager" + aten_suffix,
            ],
            visibility = ["PUBLIC"],
        )
