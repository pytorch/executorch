load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "backend" + aten_suffix,
            srcs = [
                "backend.cpp",
            ],
            exported_headers = [
                "backend.h",
            ],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = [
                "//executorch/...",
                "//executorch/test/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/core:core",
                "//executorch/core/values:executor_values" + aten_suffix,
                "//executorch/core:freeable_buffer",
                "//executorch/executor:memory_manager",
                "//executorch/profiler:profiler",
            ],
        )
