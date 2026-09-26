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
        name = "test_colorer",
        srcs = ["test_colorer.py"],
        deps = [
            "//executorch/backends/fused_quant:decompose_fused_quant",
            "//executorch/backends/fused_quant:colorer",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
        ],
    )

    runtime.python_test(
        name = "test_compile",
        srcs = ["test_compile.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/fused_quant:compile",
            "//executorch/backends/fused_quant:decompose_fused_quant",
            "//executorch/backends/fused_quant:frontend",
            "//executorch/backends/fused_quant:quantizer_defaults",
            "//executorch/backends/fused_quant:to_channels_first",
            "//executorch/exir:lib",
            "//executorch/exir:pass_manager",
        ],
    )

    runtime.python_test(
        name = "test_graph_utils",
        srcs = ["test_graph_utils.py"],
        deps = [
            "//executorch/backends/fused_quant:graph_utils",
            "//caffe2:torch",
            "//executorch/backends/test:graph_builder",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_ops",
        srcs = ["test_ops.py"],
        deps = [
            "//executorch/backends/fused_quant:ops",
            "//caffe2:torch",
        ],
    )

    runtime.python_test(
        name = "test_pass_base",
        srcs = ["test_pass_base.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:pass_base",
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
            "//executorch/backends/fused_quant:decompose_fused_quant",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fold_batch_norm",
        srcs = ["test_fold_batch_norm.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:fold_batch_norm",
            "//executorch/backends/fused_quant:replace_batch_norm_with_conv",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:compiler",
        ],
    )

    runtime.python_test(
        name = "test_fuse_add_softmax_into_masked_softmax",
        srcs = ["test_fuse_add_softmax_into_masked_softmax.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:fuse_add_softmax_into_masked_softmax",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:compiler",
        ],
    )

    runtime.python_test(
        name = "test_replace_batch_norm_with_conv",
        srcs = ["test_replace_batch_norm_with_conv.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:replace_batch_norm_with_conv",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:compiler",
        ],
    )

    runtime.python_test(
        name = "test_hoist_activations",
        srcs = ["test_hoist_activations.py"],
        deps = [
            "//executorch/backends/fused_quant:hoist_activations",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_to_channels_last",
        srcs = ["test_to_channels_last.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//executorch/backends/fused_quant:to_channels_last",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_to_channels_first",
        srcs = ["test_to_channels_first.py"],
        deps = [
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//executorch/backends/fused_quant:to_channels_first",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
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
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//executorch/backends/fused_quant:to_convolution",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
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
            "//executorch/backends/fused_quant:fuse_add_into_linear",
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_fuse_mul_into_linear",
        srcs = ["test_fuse_mul_into_linear.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//executorch/backends/fused_quant:fuse_mul_into_linear",
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_split_linear_at_slices",
        srcs = ["test_split_linear_at_slices.py"],
        deps = [
            "//executorch/backends/fused_quant:constant_fold",
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:split_linear_at_slices",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_replace_dequant_quant_with_requantize",
        srcs = ["test_replace_dequant_quant_with_requantize.py"],
        deps = [
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:replace_dequant_quant_with_requantize",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_quant_absorption",
        srcs = ["test_quant_absorption.py"],
        deps = [
            "//executorch/backends/fused_quant:graph_utils",
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:quant_absorption",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_replace_mm_with_bmm",
        srcs = ["test_replace_mm_with_bmm.py"],
        deps = [
            "//executorch/backends/fused_quant:replace_mm_with_bmm",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:compiler",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_replace_scalar_mul_with_dequant_quant",
        srcs = ["test_replace_scalar_mul_with_dequant_quant.py"],
        deps = [
            ":test_helpers",
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:replace_scalar_mul_with_dequant_quant",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:lib",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_sink_constant_cat",
        srcs = ["test_sink_constant_cat.py"],
        deps = [
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:sink_constant_cat",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/exir:lib",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_test(
        name = "test_prequantize_embedding",
        srcs = ["test_prequantize_embedding.py"],
        deps = [
            "//executorch/backends/fused_quant:ops",
            "//executorch/backends/fused_quant:prequantize_embedding",
            "//caffe2:torch",
            "//executorch/backends/cadence/aot:ops_registrations",
            "//executorch/backends/test:program_builder",
            "//executorch/exir/dialects:lib",
        ],
    )
