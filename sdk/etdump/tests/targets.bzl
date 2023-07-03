load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "etdump_gen_test",
        srcs = [
            "ETDumpGenTest.cpp",
        ],
        deps = [
            "//executorch/executor:memory_manager",
            "//executorch/profiler:profiler",
            "//executorch/sdk/etdump:etdump_gen",
        ],
        preprocessor_flags = ["-DPROFILING_ENABLED"],
    )
