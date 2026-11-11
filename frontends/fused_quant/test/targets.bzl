load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():

    runtime.python_test(
        name = "test_quantizer",
        srcs = ["test_quantizer.py"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//caffe2:torch",
            "//executorch/backends/transforms:permute_pass_utils",
            "//executorch/backends/transforms/fused_quant:graph_utils",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/frontends/fused_quant:quantizer",
            "//pytorch/ao:torchao",
        ],
    )

    runtime.python_test(
        name = "test_colorer",
        srcs = ["test_colorer.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/test:program_builder",
            "//executorch/backends/transforms/fused_quant:decompose_fused_quant",
            "//executorch/backends/transforms/fused_quant:to_channels_first",
            "//executorch/backends/transforms/fused_quant/test:test_helpers",
            "//executorch/exir:lib",
            "//executorch/exir/backend:partitioner",
            "//executorch/exir/dialects:lib",
            "//executorch/exir/dialects/edge:lib",
            "//executorch/exir/dialects/fused_quant:ops",
            "//executorch/frontends/fused_quant:colorer",
        ],
    )
