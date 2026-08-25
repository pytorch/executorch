load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")

def define_common_targets():
    if not is_fbcode():
        return

    runtime.python_library(
        name = "subclass",
        srcs = [
            "subclass.py",
        ],
        deps = [
            "//caffe2:torch",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_test(
        name = "test_subclass",
        srcs = [
            "test_subclass.py",
        ],
        deps = [
            ":subclass",
        ],
    )
