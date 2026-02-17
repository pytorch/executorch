load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Header-only library for backend options (no aten suffix needed)
    runtime.cxx_library(
        name = "backend_options",
        exported_headers = [
            "options.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    # Header-only library for backend options map (no aten suffix needed)
    runtime.cxx_library(
        name = "backend_options_map",
        exported_headers = [
            "backend_options_map.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":backend_options",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "interface" + aten_suffix,
            srcs = [
                "interface.cpp",
            ],
            exported_headers = [
                "backend_execution_context.h",
                "backend_init_context.h",
                "backend_option_context.h",
                "backend_options_map.h",
                "options.h",
                "interface.h",
            ],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                "//executorch/runtime/core:memory_allocator",
                "//executorch/runtime/core:named_data_map",
            ],
        )
