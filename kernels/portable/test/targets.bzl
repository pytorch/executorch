load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib", "op_test")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    define_supported_features_lib()

    op_test(name = "op__to_dim_order_copy_test", kernel_name = "aten", deps = [
        "//executorch/kernels/portable/cpu:op__to_dim_order_copy_aten",
        "//executorch/kernels/portable:generated_lib_edge_dialect_ops_aten_headers",
    ])
    op_test(name = "op__to_dim_order_copy_test", is_edge_dialect_op = True)
    op_test(name = "op_allclose_test")
    op_test(name = "op_div_test")
    op_test(name = "op_gelu_test")
    op_test(name = "op_mul_test")
