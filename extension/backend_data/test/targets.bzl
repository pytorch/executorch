load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets shared between fbcode and xplat."""

    runtime.cxx_test(
        name = "file_data_writer_test",
        srcs = ["file_data_writer_test.cpp"],
        deps = [
            "//executorch/extension/backend_data:buffer_data_writer",
            "//executorch/extension/backend_data:file_data_writer",
        ],
    )

    runtime.cxx_test(
        name = "initialize_and_save_test",
        srcs = ["initialize_and_save_test.cpp"],
        deps = [
            "//executorch/extension/backend_data:buffer_data_writer",
            "//executorch/extension/backend_data:initialize_and_save",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/executor:program",
            "//executorch/schema:extended_header",
            "//executorch/schema:program",
        ],
    )
