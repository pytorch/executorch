load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "optimizer" + aten_suffix,
            exported_headers = [
                "sgd.h",
            ],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )
