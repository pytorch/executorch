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
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )
