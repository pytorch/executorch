load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "print_evalue" + aten_suffix,
            srcs = ["print_evalue.cpp"],
            exported_headers = ["print_evalue.h"],
            visibility = ["@EXECUTORCH_CLIENTS"],
            exported_deps = [
                "//executorch/runtime/core:evalue" + aten_suffix,
            ],
            deps = [
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
        )
