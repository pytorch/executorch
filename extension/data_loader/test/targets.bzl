load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "buffer_data_loader_test",
        srcs = [
            "buffer_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/extension/data_loader:buffer_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "shared_ptr_data_loader_test",
        srcs = [
            "shared_ptr_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/extension/data_loader:shared_ptr_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "file_data_loader_test",
        srcs = [
            "file_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/extension/testing_util:temp_file",
            "//executorch/extension/data_loader:file_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "mmap_data_loader_test",
        srcs = [
            "mmap_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/extension/testing_util:temp_file",
            "//executorch/extension/data_loader:mmap_data_loader",
        ],
    )
