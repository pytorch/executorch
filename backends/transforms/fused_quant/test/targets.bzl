load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

_VISIBILITY = ["PUBLIC"]

def define_common_targets():

    runtime.python_library(
        name = "test_helpers",
        srcs = ["helpers.py"],
        visibility = _VISIBILITY,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_test(
        name = "test_pass_base",
        srcs = ["test_pass_base.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:pass_base",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_decompose_fused_quant",
        srcs = ["test_decompose_fused_quant.py"],
        deps = [
            "//executorch/frontends/fused_quant:colorer",
            "//executorch/backends/transforms/fused_quant:decompose_fused_quant",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fold_batch_norm",
        srcs = ["test_fold_batch_norm.py"],
        deps = [
            ":test_helpers",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:fold_batch_norm",
            "//executorch/backends/transforms/fused_quant:replace_batch_norm_with_conv",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_fuse_add_softmax_into_masked_softmax",
        srcs = ["test_fuse_add_softmax_into_masked_softmax.py"],
        deps = [
            ":test_helpers",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:fuse_add_softmax_into_masked_softmax",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_replace_batch_norm_with_conv",
        srcs = ["test_replace_batch_norm_with_conv.py"],
        deps = [
            ":test_helpers",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:replace_batch_norm_with_conv",
            "//caffe2:torch",
            "//executorch/backends/transforms:utils",
        ],
    )

    runtime.python_test(
        name = "test_fold_input_free_subgraphs",
        srcs = ["test_fold_input_free_subgraphs.py"],
        deps = [
            ":test_helpers",
            "//executorch/backends/transforms/fused_quant:fold_input_free_subgraphs",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_replace_adaptive_avg_pool_with_avg_pool",
        srcs = ["test_replace_adaptive_avg_pool_with_avg_pool.py"],
        deps = [
            ":test_helpers",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:replace_adaptive_avg_pool_with_avg_pool",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_replace_clamp_with_hardtanh",
        srcs = ["test_replace_clamp_with_hardtanh.py"],
        deps = [
            ":test_helpers",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:replace_clamp_with_hardtanh",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_replace_sub_constant_with_add",
        srcs = ["test_replace_sub_constant_with_add.py"],
        deps = [
            ":test_helpers",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:replace_sub_constant_with_add",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_hoist_activations",
        srcs = ["test_hoist_activations.py"],
        deps = [
            "//executorch/backends/transforms/fused_quant:hoist_activations",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_to_channels_last",
        srcs = ["test_to_channels_last.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//executorch/frontends/fused_quant:colorer",
            "//executorch/backends/transforms/fused_quant:to_channels_last",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_to_channels_first",
        srcs = ["test_to_channels_first.py"],
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//executorch/frontends/fused_quant:colorer",
            "//executorch/backends/transforms/fused_quant:to_channels_first",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_to_convolution",
        srcs = ["test_to_convolution.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//executorch/backends/transforms/fused_quant:to_convolution",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fuse_add_into_linear",
        srcs = ["test_fuse_add_into_linear.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:fuse_add_into_linear",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fuse_mul_into_linear",
        srcs = ["test_fuse_mul_into_linear.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:fuse_mul_into_linear",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_canonicalize_linear_input_rank",
        srcs = ["test_canonicalize_linear_input_rank.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:canonicalize_linear_input_rank",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_split_linear_at_slices",
        srcs = ["test_split_linear_at_slices.py"],
        deps = [
            "//executorch/backends/transforms/fused_quant:constant_fold",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:split_linear_at_slices",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_replace_dequant_quant_with_requantize",
        srcs = ["test_replace_dequant_quant_with_requantize.py"],
        deps = [
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:replace_dequant_quant_with_requantize",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_quant_absorption",
        srcs = ["test_quant_absorption.py"],
        deps = [
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:quant_absorption",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms:utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fold_scalar_mul",
        srcs = ["test_fold_scalar_mul.py"],
        deps = [
            ":test_helpers",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:fold_scalar_mul",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:lib",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_sink_constant_cat",
        srcs = ["test_sink_constant_cat.py"],
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:sink_constant_cat",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:lib",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_prequantize_embedding",
        srcs = ["test_prequantize_embedding.py"],
        deps = [
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:prequantize_embedding",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_op_numerics",
        srcs = ["test_op_numerics.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/transforms/fused_quant:decompose_fused_quant",
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/backends/transforms/fused_quant:to_channels_first",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:pass_manager",
            "//executorch/exir/dialects:lib",
        ],
    )
