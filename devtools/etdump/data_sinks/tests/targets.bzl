load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_data_sink_test(data_sink_name):
    runtime.cxx_test(
        name = data_sink_name + "_test",
        srcs = [
            data_sink_name + "_test.cpp",
        ],
        deps = [
            "//executorch/devtools/etdump/data_sinks:" + data_sink_name,
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    define_data_sink_test("buffer_data_sink")
    define_data_sink_test("file_data_sink")
