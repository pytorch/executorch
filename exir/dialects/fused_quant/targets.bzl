load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# fused_quant is a backend-neutral quantization frontend, so it is consumed both
# from inside ExecuTorch and from downstream compilers built on top of it.
_VISIBILITY = ["PUBLIC"]

def define_common_targets():
    runtime.python_library(
        name = "ops_utils",
        srcs = ["ops_utils.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "ops",
        srcs = ["ops.py"],
        visibility = _VISIBILITY,
        deps = [
            ":ops_utils",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//pytorch/ao:torchao",
        ],
    )
