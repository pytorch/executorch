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
            "//executorch/backends/transforms/fused_quant:decompose_fused_quant",
            "//executorch/exir/dialects/fused_quant:ops",
            ":test_helpers",
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/exir/dialects:lib",
        ],
    )
