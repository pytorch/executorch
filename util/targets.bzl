load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "system",
        exported_headers = [
            "system.h",
        ],
        visibility = [
            "//executorch/util/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "test_memory_config",
        srcs = [],
        exported_headers = ["TestMemoryConfig.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:memory_allocator",
        ],
    )

    runtime.cxx_library(
        name = "read_file",
        srcs = ["read_file.cpp"],
        exported_headers = ["read_file.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            ":system",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:compiler",
        ],
    )

    runtime.cxx_library(
        name = "memory_utils",
        srcs = ["memory_utils.cpp"],
        exported_headers = ["memory_utils.h"],
        visibility = [
            "//executorch/backends/...",
            "//executorch/runtime/backend/...",
            "//executorch/util/test/...",
        ],
        deps = [
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            ":system",
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
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
                "//executorch/runtime/executor:executor" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/schema:schema",
                "//executorch/schema:bundled_program_schema",
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
                "//executorch/runtime/executor:executor" + aten_suffix,
            ],
        )
