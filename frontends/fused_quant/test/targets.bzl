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
