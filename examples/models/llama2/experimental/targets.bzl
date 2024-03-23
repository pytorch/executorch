load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
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
