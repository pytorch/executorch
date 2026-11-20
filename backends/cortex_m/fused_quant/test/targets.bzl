load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_test(
        name = "test_lowering",
        srcs = ["test_lowering.py"],
        typing = True,
        preload_deps = ["//executorch/kernels/quantized:aot_lib"],
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "//caffe2:torch",
            "//executorch/backends/cortex_m/fused_quant:fused_quant",
            "//executorch/backends/transforms/fused_quant:passes",
            "//executorch/backends/cortex_m/quantizer:quantization_configs",
            "//executorch/exir:pass_manager",
            "//executorch/exir/dialects:lib",
            "//executorch/export:lib",
            "//executorch/frontends/fused_quant:quantizer",
            "//executorch/frontends/fused_quant:recipe",
            "//pytorch/ao:torchao",
        ],
    )
