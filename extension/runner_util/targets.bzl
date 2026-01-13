load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "inputs" + aten_suffix,
            srcs = [
                "inputs.cpp",
                "inputs{}.cpp".format("_aten" if aten_mode else "_portable"),
            ],
            exported_headers = ["inputs.h"],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/executor:program_no_prim_ops" + aten_suffix,
            ],
        )
