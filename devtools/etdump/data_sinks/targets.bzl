load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "data_sink_base" + aten_suffix,
            exported_headers = [
                "data_sink_base.h",
            ],
            exported_deps = [
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

        runtime.cxx_library(
            name = "buffer_data_sink" + aten_suffix,
            exported_headers = [
                "buffer_data_sink.h",
            ],
            srcs = [
                "buffer_data_sink.cpp",
            ],
            deps = [
                "//executorch/devtools/etdump:utils",
            ],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                ":data_sink_base" + aten_suffix,
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )
