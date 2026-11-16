load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

_VISIBILITY = ["PUBLIC"]

def define_common_targets():
    runtime.python_library(
        name = "colorer",
        srcs = ["colorer.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_library(
        name = "quantizer",
        srcs = [
            "quantizer/__init__.py",
            "quantizer/quantizer.py",
        ],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/backends/transforms/fused_quant:fuse_aten",
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "recipe",
        srcs = ["recipe.py"],
        visibility = _VISIBILITY,
        deps = [
            ":quantizer",
            "//caffe2:torch",
            "//executorch/backends/transforms/fused_quant:decompose_fused_quant",
            "//executorch/backends/transforms/fused_quant:passes",
            "//executorch/backends/transforms/fused_quant:to_channels_first",
            "//executorch/exir:_warnings",
            "//executorch/exir:lib",
            "//executorch/exir:pass_manager",
            "//executorch/exir/dialects/edge:lib",
            "//executorch/export:recipe",
            "//executorch/export:types",
            "//executorch/kernels/quantized:quantized_ops_lib",
            "//pytorch/ao:torchao",
        ],
    )
