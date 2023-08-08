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
        visibility = [
            "//executorch/exir/backend/test/...",
            "//executorch/runtime/executor/test/...",
            "//executorch/extension/pybindings/...",
            "//executorch/test/...",
            "//executorch/extension/data_loader/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "shared_ptr_data_loader",
        srcs = [],
        exported_headers = ["shared_ptr_data_loader.h"],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//executorch/extension/data_loader/test/...",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "file_data_loader",
        srcs = ["file_data_loader.cpp"],
        exported_headers = ["file_data_loader.h"],
        visibility = [
            "//executorch/test/...",
            "//executorch/runtime/executor/test/...",
            "//executorch/extension/data_loader/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "mmap_data_loader",
        srcs = ["mmap_data_loader.cpp"],
        exported_headers = ["mmap_data_loader.h"],
        visibility = [
            "//executorch/test/...",
            "//executorch/extension/pybindings/...",
            "//executorch/runtime/executor/test/...",
            "//executorch/extension/data_loader/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )
