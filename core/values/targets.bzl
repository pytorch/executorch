load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "executor_values" + aten_suffix,
            exported_headers = [
                "Evalue.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/core:core",
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
                ":executor_tag",
            ],
        )

    runtime.cxx_library(
        name = "executor_tag",
        exported_headers = [
            "Tag.h",
        ],
        visibility = [
            "//executorch/...",
        ],
    )
