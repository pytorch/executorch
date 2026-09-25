load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets shared between fbcode and xplat."""

    runtime.cxx_library(
        name = "data_writer",
        srcs = [],
        exported_headers = ["data_writer.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "file_data_writer",
        srcs = ["file_data_writer.cpp"],
        exported_headers = ["file_data_writer.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":data_writer",
        ],
    )

    runtime.cxx_library(
        name = "buffer_data_writer",
        srcs = ["buffer_data_writer.cpp"],
        exported_headers = ["buffer_data_writer.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":data_writer",
        ],
    )

    runtime.cxx_library(
        name = "initialize_and_save",
        srcs = ["initialize_and_save.cpp"],
        exported_headers = ["initialize_and_save.h"],
        visibility = ["PUBLIC"],
        deps = [
            ":data_writer",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/executor:program",
            "//executorch/schema:extended_header",
            "//executorch/schema:program",
        ],
    )
