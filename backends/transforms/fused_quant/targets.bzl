load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# fused_quant is a backend-neutral quantization frontend, so it is consumed both
# from inside ExecuTorch and from downstream compilers built on top of it.
_VISIBILITY = ["PUBLIC"]

def define_common_targets():
    runtime.python_library(
        name = "graph_utils",
        srcs = ["graph_utils.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects/edge:lib",
        ],
    )

    runtime.python_library(
        name = "pass_base",
        srcs = ["pass_base.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir:pass_base",
            "//executorch/exir:pass_manager",
        ],
    )

    runtime.python_library(
        name = "fuse_aten",
        srcs = ["fuse_aten.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
        ],
    )

    runtime.python_library(
        name = "decompose_fused_quant",
        srcs = ["decompose_fused_quant.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
        ],
    )
