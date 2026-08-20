# custom_kernel_example

This directory shows an example about how to use the kernel library framework to automatically test an ATen-compliant operator from a custom kernel.

- my_functions.yaml: contains the op for my custom kernel. Contains only relu for now.
- op_relu.cpp: the impl for my custom kernel. The code is copied from portable kernel.
- targets.bzl: defines the op library and the ET codegen generated lib for the kernel.
- tests.bzl: defines the new targets needed to invoke tests from kernel library framework to test my relu implementation automatically.

## What's tests.bzl?

To clarify, we don't need a separate bzl file for it. We can use targets.bzl and add targets there directly. Here it's for demo purpose.

We need to invoke `generated_op_test` from executorch/kernels/test/util.bzl so we can test automatically.

Follow steps in tests.bzl to define required targets.

- Step 1: Define the function header wrapper in executorch/kernels/test/targets.bzl, like
  `codegen_function_header_wrapper("executorch/kernels/test/custom_kernel_example", "custom_kernel_example")`
  or generally `codegen_function_header_wrapper("<path-to-your-kernel>/<your-kernel-name>", "<your-kernel-name>")`
  This is needed because tests need to know our Functions.h target.
  TODO(T149423767): We should codegen this wrapper in #include, not let user define it.

- Step 2: Use the helper to produce the supported feature list for tests.

  We need an additional supported_features_def.yaml for test codegen. See examples in executorch/kernels/test/supported_features.yaml and supported_features_def_example.yaml.
  Need to override some default features if different.

   ```
   define_supported_features_lib()
   ```

- Step 3: Use the helper generated_op_test to re-use existing tests
  ```
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
  ```
