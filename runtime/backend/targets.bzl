load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "interface" + aten_suffix,
            srcs = [
                "interface.cpp",
            ],
            exported_headers = [
                "backend_execution_context.h",
                "backend_init_context.h",
                "interface.h",
            ],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                "//executorch/runtime/core:memory_allocator",
            ],
        )
