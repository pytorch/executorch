load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():

    runtime.python_test(
        name = "test_ops",
        srcs = ["test_ops.py"],
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
        ],
    )
