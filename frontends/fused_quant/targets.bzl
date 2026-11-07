load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

_VISIBILITY = ["PUBLIC"]

def define_common_targets():
    runtime.python_library(
        name = "quantizer",
        srcs = [
            "quantizer/__init__.py",
            "quantizer/quantizer.py",
        ],
        visibility = _VISIBILITY,
        deps = [
            ":qspecs",
            "//executorch/backends/transforms/fused_quant:fuse_aten",
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "qspecs",
        srcs = ["quantizer/qspecs.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//pytorch/ao:torchao",
        ],
    )
