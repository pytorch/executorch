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

        # Attention Sink IOManager for runner-side infinite context
        runtime.cxx_library(
            name = "attention_sink_io_manager" + aten_suffix,
            srcs = [
                "attention_sink_io_manager.cpp",
            ],
            exported_headers = [
                "attention_sink_io_manager.h",
            ],
            exported_deps = [
                ":io_manager" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
            ],
            visibility = ["PUBLIC"],
        )
