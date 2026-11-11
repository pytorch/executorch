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
            "//executorch/frontends/fused_quant:colorer",
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

    runtime.python_library(
        name = "fold_batch_norm",
        srcs = ["pre_quantize_passes/fold_batch_norm.py"],
        visibility = _VISIBILITY,
        deps = [
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "fuse_add_softmax_into_masked_softmax",
        srcs = ["pre_quantize_passes/fuse_add_softmax_into_masked_softmax.py"],
        visibility = _VISIBILITY,
        deps = [
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "replace_batch_norm_with_conv",
        srcs = ["pre_quantize_passes/replace_batch_norm_with_conv.py"],
        visibility = _VISIBILITY,
        deps = [
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "fold_input_free_subgraphs",
        srcs = ["pre_quantize_passes/fold_input_free_subgraphs.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_library(
        name = "replace_adaptive_avg_pool_with_avg_pool",
        srcs = ["pre_quantize_passes/replace_adaptive_avg_pool_with_avg_pool.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
        ],
    )

    runtime.python_library(
        name = "replace_clamp_with_hardtanh",
        srcs = ["pre_quantize_passes/replace_clamp_with_hardtanh.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
        ],
    )

    runtime.python_library(
        name = "constant_fold",
        srcs = ["optimization_passes/constant_fold.py"],
        visibility = _VISIBILITY,
        deps = [
            ":pass_base",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "hoist_activations",
        srcs = ["optimization_passes/hoist_activations.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "to_channels_last",
        srcs = ["optimization_passes/to_channels_last.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/frontends/fused_quant:colorer",
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/exir/dialects/fused_quant:ops_utils",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "to_channels_first",
        srcs = ["optimization_passes/to_channels_first.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/frontends/fused_quant:colorer",
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "to_convolution",
        srcs = ["optimization_passes/to_convolution.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "fuse_add_into_linear",
        srcs = ["optimization_passes/fuse_add_into_linear.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "fuse_mul_into_linear",
        srcs = ["optimization_passes/fuse_mul_into_linear.py"],
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
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "canonicalize_linear_input_rank",
        srcs = ["optimization_passes/canonicalize_linear_input_rank.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "split_linear_at_slices",
        srcs = ["optimization_passes/split_linear_at_slices.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "replace_sub_constant_with_add",
        srcs = ["optimization_passes/replace_sub_constant_with_add.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "fold_scalar_mul",
        srcs = ["optimization_passes/fold_scalar_mul.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "sink_constant_cat",
        srcs = ["optimization_passes/sink_constant_cat.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "prequantize_embedding",
        srcs = ["optimization_passes/prequantize_embedding.py"],
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
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "quant_absorption",
        srcs = ["optimization_passes/quant_absorption.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
            "//executorch/exir/passes:constant_prop_pass",
        ],
    )

    runtime.python_library(
        name = "replace_dequant_quant_with_requantize",
        srcs = ["optimization_passes/replace_dequant_quant_with_requantize.py"],
        visibility = _VISIBILITY,
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )
