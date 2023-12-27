load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "runner",
        srcs = [
            "runner.cpp",
        ],
        exported_headers = [
            "runner.h",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/executor:program",
        ],
    )

    runtime.cxx_library(
        name = "runner_aten",
        srcs = [
            "runner.cpp",
        ],
        exported_headers = [
            "runner.h",
        ],
        platforms = ["Default"],
        define_static_target = False,
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/executor:program_aten",
        ],
    )
