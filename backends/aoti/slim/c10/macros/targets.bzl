load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor c10 macros module."""

    # Header-only library for Macros
    runtime.cxx_library(
        name = "macros",
        headers = [
            "Macros.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
    )
