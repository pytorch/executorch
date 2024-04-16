load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib", "op_test")

def define_common_targets():
    define_supported_features_lib()

    op_test("op_quantize_test", kernel_name = "quantized")
    op_test("op_dequantize_test", kernel_name = "quantized")
    op_test("op_choose_qparams_test", kernel_name = "quantized")
    op_test("op_add_test", kernel_name = "quantized", deps = [
        "//executorch/kernels/quantized/cpu:op_dequantize",
        "//executorch/kernels/quantized/cpu:op_quantize",
        "//executorch/kernels/quantized/cpu:op_add",
        "//executorch/kernels/quantized:generated_lib_headers",
        "//executorch/kernels/portable:generated_lib_headers",
        "//executorch/kernels/portable/cpu:op_add",
        "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
    ])
    op_test("op_embedding_test", kernel_name = "quantized", deps = [
        "//executorch/kernels/quantized/cpu:op_dequantize",
        "//executorch/kernels/quantized/cpu:op_quantize",
        "//executorch/kernels/quantized/cpu:op_add",
        "//executorch/kernels/quantized/cpu:op_embedding",
        "//executorch/kernels/quantized:generated_lib_headers",
        "//executorch/kernels/portable:generated_lib_headers",
        "//executorch/kernels/portable/cpu:op_embedding",
        "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
    ])
    op_test("op_embedding4b_test", kernel_name = "quantized")
    op_test("op_mixed_mm_test", kernel_name = "quantized", deps = [
        "//executorch/kernels/quantized/cpu:op_mixed_mm",
        "//executorch/kernels/quantized:generated_lib_headers",
        "//executorch/kernels/portable:generated_lib_headers",
        "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
    ])
    op_test("op_mixed_linear_test", kernel_name = "quantized", deps = [
        "//executorch/kernels/quantized/cpu:op_mixed_linear",
        "//executorch/kernels/quantized:generated_lib_headers",
        "//executorch/kernels/portable:generated_lib_headers",
        "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
    ])
