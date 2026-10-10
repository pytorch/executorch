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

    runtime.python_library(
        name = "graph_utils",
        srcs = ["graph_utils.py"],
        visibility = _VISIBILITY,
        deps = [
            ":ops",
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
            ":ops",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
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
            ":replace_mm_with_bmm",
            ":fuse_aten",
            ":graph_utils",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "colorer",
        srcs = ["colorer.py"],
        visibility = _VISIBILITY,
        deps = [
            ":pass_base",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects/edge:lib",
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

    runtime.python_library(
        name = "quantizer_defaults",
        srcs = ["quantizer/defaults.py"],
        visibility = _VISIBILITY,
        deps = [
            ":qspecs",
            ":quantizer",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "frontend",
        srcs = ["frontend.py"],
        visibility = _VISIBILITY,
        deps = [
            ":fold_batch_norm",
            ":fuse_add_softmax_into_masked_softmax",
            ":quantizer",
            ":replace_batch_norm_with_conv",
            "//caffe2:torch",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "passes",
        srcs = ["passes.py"],
        visibility = _VISIBILITY,
        deps = [
            ":constant_fold",
            ":fold_scalar_mul",
            ":fuse_add_into_linear",
            ":fuse_mul_into_linear",
            ":graph_utils",
            ":hoist_activations",
            ":ops",
            ":pass_base",
            ":prequantize_embedding",
            ":quant_absorption",
            ":replace_dequant_quant_with_requantize",
            ":sink_constant_cat",
            ":split_linear_at_slices",
            ":to_channels_last",
            ":to_convolution",
            "//caffe2:torch",
            "//executorch/backends/transforms:advance_quantize_op_above_def_chain",
            "//executorch/backends/transforms:advance_quantize_op_above_def_in_branch",
            "//executorch/backends/transforms:fuse_cascaded_transpose_or_permute_ops",
            "//executorch/backends/transforms:fuse_cascaded_view_ops",
            "//executorch/backends/transforms:fuse_quant_dequant_to_requantize",
            "//executorch/backends/transforms:merge_split_concat_chain",
            "//executorch/backends/transforms:move_permute_after_concat",
            "//executorch/backends/transforms:move_slice_before_permute",
            "//executorch/backends/transforms:move_slice_before_view",
            "//executorch/backends/transforms:postpone_dequantize_op_below_use_chain",
            "//executorch/backends/transforms:propagate_slice",
            "//executorch/backends/transforms:remove_alias_copy_op",
            "//executorch/backends/transforms:remove_branched_quant_dequant",
            "//executorch/backends/transforms:remove_cat_from_slice_copy",
            "//executorch/backends/transforms:remove_clone_ops_transform_imported",
            "//executorch/backends/transforms:remove_nop_as_strided_copy_op",
            "//executorch/backends/transforms:remove_nop_expand_op",
            "//executorch/backends/transforms:remove_nop_slice_or_view_op",
            "//executorch/backends/transforms:remove_permute_before_mean",
            "//executorch/backends/transforms:remove_permutes_around_elementwise_ops",
            "//executorch/backends/transforms:remove_zero_sized_cat_args",
            "//executorch/backends/transforms:replace_nop_transpose_or_permute_with_view",
            "//executorch/backends/transforms:replace_select_with_view_op",
            "//executorch/backends/transforms:replace_split_with_slice",
            "//executorch/backends/transforms:replace_squeeze_unsqueeze_with_view",
            "//executorch/backends/transforms:split_dequantized_cat",
            "//executorch/exir:pass_manager",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
        ],
    )

    runtime.python_library(
        name = "compile",
        srcs = ["compile.py"],
        visibility = _VISIBILITY,
        deps = [
            ":frontend",
            ":passes",
            ":quantizer",
            ":quantizer_defaults",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/exir:pass_manager",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_library(
        name = "decompose_fused_quant",
        srcs = ["decompose_fused_quant.py"],
        visibility = _VISIBILITY,
        deps = [
            ":colorer",
            ":graph_utils",
            ":ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
        ],
    )

    # ------------------------------------------------------------------
    # Pre-quantization passes: rewrites applied to the traced ExportedProgram
    # before the quantizer annotates it.
    # ------------------------------------------------------------------

    runtime.python_library(
        name = "replace_mm_with_bmm",
        srcs = ["pre_quantize_passes/replace_mm_with_bmm.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
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

    # ------------------------------------------------------------------
    # Optimization passes: rewrites applied to the fused_quant edge graph.
    # ------------------------------------------------------------------

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
            ":ops",
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
            ":colorer",
            ":graph_utils",
            ":ops",
            ":ops_utils",
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
            ":colorer",
            ":graph_utils",
            ":ops",
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
            ":ops",
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
            ":ops",
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
        name = "split_linear_at_slices",
        srcs = ["optimization_passes/split_linear_at_slices.py"],
        visibility = _VISIBILITY,
        deps = [
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
        name = "fold_scalar_mul",
        srcs = ["optimization_passes/fold_scalar_mul.py"],
        visibility = _VISIBILITY,
        deps = [
            ":graph_utils",
            ":ops",
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
            ":ops",
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
            ":ops",
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
            ":ops",
            ":pass_base",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )
