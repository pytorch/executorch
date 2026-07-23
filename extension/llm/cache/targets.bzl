load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    runtime.python_library(
        name = "cache",
        srcs = [
            "reference_cache.py",
            "update_and_attend.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
        ],
    )
