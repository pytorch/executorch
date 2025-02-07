load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "training_module" + aten_suffix,
            srcs = [
                "training_module.cpp",
            ],
            exported_headers = [
                "training_module.h",
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
            ],
        )
