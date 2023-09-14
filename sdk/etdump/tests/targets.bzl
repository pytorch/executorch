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
            "//executorch/runtime/core:memory_allocator",
            "//executorch/sdk/etdump:etdump_gen",
        ],
        preprocessor_flags = ["-DPROFILING_ENABLED"],
    )

    runtime.cxx_test(
        name = "etdump_test",
        srcs = [
            "etdump_test.cpp",
        ],
        deps = [
            "//executorch/sdk/etdump:etdump_flatcc",
            "//executorch/sdk/etdump:etdump_schema_flatcc",
            "//executorch/runtime/platform:platform",
        ],
    )
