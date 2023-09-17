load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "read_file",
        srcs = ["read_file.cpp"],
        exported_headers = ["read_file.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:compiler",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "bundled_program_verification" + aten_suffix,
            srcs = ["bundled_program_verification.cpp"],
            exported_headers = ["bundled_program_verification.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/schema:bundled_program_schema",
                "//executorch/schema:program",
            ],
            exported_deps = [
                "//executorch/runtime/core:memory_allocator",
                "//executorch/runtime/executor:program" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "util" + aten_suffix,
            srcs = [],
            exported_headers = ["util.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/executor:program" + aten_suffix,
                "//executorch/runtime/platform:platform",
            ],
        )
