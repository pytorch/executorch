load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "etdump_test",
        srcs = [
            "etdump_test.cpp",
        ],
        deps = [
            "//executorch/devtools/etdump:etdump_flatcc",
            "//executorch/devtools/etdump:etdump_schema_flatcc",
            "//executorch/devtools/etdump/data_sinks:file_data_sink",
            "//executorch/extension/testing_util:temp_file",
            "//executorch/runtime/platform:platform",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "etdump_filter_test",
        srcs = [
            "etdump_filter_test.cpp",
        ],
        deps = [
            "//executorch/devtools/etdump:etdump_filter",
            "//executorch/runtime/platform:platform",
        ],
    )
