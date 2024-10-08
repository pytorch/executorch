load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "dumb_fht",
        srcs = ["dumb_fht.c"],
        exported_headers = ["dumb_fht.h"],
        visibility = ["@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "fht",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": ["fht_neon.c"],
            "ovr_config//cpu:x86_64": ["fht_avx.c"],
        }),
        exported_headers = ["fht.h"],
        visibility = ["@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_binary(
        name = "test_float",
        srcs = ["test_float.c"],
        deps = [
            ":dumb_fht",
            ":fht",
        ],
    )
