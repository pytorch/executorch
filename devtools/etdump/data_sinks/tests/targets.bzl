load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """


    runtime.cxx_test(
        name = "buffer_data_sink_test",
        srcs = [
            "buffer_data_sink_test.cpp",
        ],
        deps = [
            "//executorch/devtools/etdump/data_sinks:buffer_data_sink",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )


    runtime.cxx_test(
        name = "stream_data_sink_test",
        srcs = [
            "stream_data_sink_test.cpp",
        ],
        deps = [
            "//executorch/devtools/etdump/data_sinks:stream_data_sink",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
