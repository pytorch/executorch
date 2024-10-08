load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "lib",
        srcs = [
            "__init__.py",
        ],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            ":addmm_mm_to_linear",
        ],
    )

    runtime.python_library(
        name = "addmm_mm_to_linear",
        srcs = ["addmm_mm_to_linear.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir:sym_util",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "decompose_sdpa",
        srcs = ["decompose_sdpa.py"],
        visibility = [
            "//executorch/backends/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_library(
        name = "fuse_batch_norm_with_conv",
        srcs = ["fuse_batch_norm_with_conv.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            ":utils",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir:sym_util",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "fuse_conv_with_clamp",
        srcs = ["fuse_conv_with_clamp.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            ":utils",
            "//caffe2:torch",
            "//executorch/backends/vulkan/passes:custom_ops_defs",
            "//executorch/exir:pass_base",
            "//executorch/exir:sym_util",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "fuse_dequant_linear",
        srcs = ["fuse_dequant_linear.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            ":utils",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir:sym_util",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "view_copy_to_squeeze_unsqueeze",
        srcs = ["view_copy_to_squeeze_unsqueeze.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            ":utils",
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "fuse_view_copy",
        srcs = ["fuse_view_copy.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "remove_clone_ops",
        srcs = ["remove_clone_ops.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "mean_to_sum_div",
        srcs = ["mean_to_sum_div.py"],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
            "//executorch/exir:sym_util",
            "//executorch/exir/dialects:lib",
        ],
    )

    runtime.python_library(
        name = "utils",
        srcs = ["utils.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/exir:pass_manager",
            "//executorch/exir/backend/canonical_partitioners:canonical_partitioner_lib",
            "//executorch/exir/dialects:lib",
            "//pytorch/ao:torchao",  # @manual
        ],
    )

    runtime.python_library(
        name = "duplicate_dynamic_quant_chain",
        srcs = ["duplicate_dynamic_quant_chain.py"],
        visibility = [
            "//executorch/backends/...",
            "//executorch/examples/...",
            "//executorch/extension/llm/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "convert_dtype_pass",
        srcs = [
            "convert_dtype_pass.py",
        ],
        visibility = [
            "//executorch/backends/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_test(
        name = "test_duplicate_dynamic_quant_chain",
        srcs = [
            "test/test_duplicate_dynamic_quant_chain.py",
        ],
        deps = [
            "fbsource//third-party/pypi/expecttest:expecttest",  # @manual
            ":duplicate_dynamic_quant_chain",
            "//caffe2:torch",
            "//executorch/exir:lib",
        ],
    )
