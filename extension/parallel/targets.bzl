load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "thread_parallel" + aten_suffix,
            srcs = [
                "thread_parallel.cpp",
            ],
            exported_headers = [
                "thread_parallel.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/extension/threadpool:threadpool",
                "//executorch/runtime/core:core",
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
        )
