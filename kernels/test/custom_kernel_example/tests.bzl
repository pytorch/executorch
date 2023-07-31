load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib", "generated_op_test")
load(":targets.bzl", "MY_ATEN_COMPLIANT_OPS")

def define_common_test_targets():
    # Step 1: Define the function header wrapper in executorch/kernels/test/targets.bzl, like
    # `codegen_function_header_wrapper("executorch/kernels/test/custom_kernel_example", "custom_kernel_example")`
    # or generally `codegen_function_header_wrapper("<path-to-your-kernel>/<your-kernel-name>", "<your-kernel-name>")`
    # This is needed because tests need to know our Functions.h target.
    # TODO(T149423767): We should codegen this wrapper in #include, not let user define it.

    # Step 2: Use the helper to produce the supported feature list for tests.
    # Need to override some default features if different.
    # See executorch/kernels/test/supported_features.yaml and supported_features_def_example.yaml.
    define_supported_features_lib()

    # Step 3: Use the helper generated_op_test to re-use existing tests
    for op in MY_ATEN_COMPLIANT_OPS:
        op_name = op["name"]

        generated_op_test(
            name = op_name + "_test",
            op_impl_target = ":my_operators",
            generated_lib_headers_target = ":generated_lib_headers",

            # those two targets are defined in previous steps
            supported_features_target = ":supported_features",
            function_header_wrapper_target = "//executorch/kernels/test:function_header_wrapper_custom_kernel_example",
        )
