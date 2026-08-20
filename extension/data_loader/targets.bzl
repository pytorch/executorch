load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "buffer_data_loader",
        srcs = [],
        exported_headers = ["buffer_data_loader.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "shared_ptr_data_loader",
        srcs = [],
        exported_headers = ["shared_ptr_data_loader.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "file_data_loader",
        srcs = ["file_data_loader.cpp"],
        exported_headers = ["file_data_loader.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "file_descriptor_data_loader",
        srcs = ["file_descriptor_data_loader.cpp"],
        exported_headers = ["file_descriptor_data_loader.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "mmap_data_loader",
        srcs = [
            "mmap_data_loader.cpp"
        ] + select({
            "DEFAULT": [],
            "ovr_config//os:windows": ["mman_windows.cpp"],
        }),
        headers = select({
            "DEFAULT": [],
            "ovr_config//os:windows": ["mman_windows.h"],
        }),
        exported_headers = [
            "mman.h",
            "mmap_data_loader.h"
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )
