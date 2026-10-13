# ExecuTorch provider

[`ETProvider`](ETProvider.cpp) maps supported semantic operations to registered
ExecuTorch out kernels. Its implementation is named `ET/registry`. Tensor dtypes
map by name to ATen scalar types; quantized tensors are rejected because ET
routes do not interpret quantization metadata. The route table covers the
linked portable kernel set; factory kernels stack a subset of their inputs
because the kernels read dtype and shape from the output tensor.

The provider resolves registered kernels when it is created and keeps a sorted
table of linked routes for selection and compilation. Kernel registration must
finish before provider creation. Selection and compilation share argument and
tensor validation; dynamic arguments and incomplete tensor lists are rejected
during selection. Static floating-point lists are supported for upsampling.
Integer literals in `float` and `float?` schema positions are boxed as doubles;
`Scalar` arguments preserve their numeric type.

Each executable owns its boxed argument frames and tensor metadata for only the
values it references. Repeated tensor arguments share an entry within that
executable. The shared
[`KernelProvider`](../../KernelProvider.h) interface exposes Native graph metadata
and buffers. Current routes require no temporary allocation; a new route that
needs it must declare storage requirements and use planned scratch.

The `cpu_et` target in [`targets.bzl`](../../../targets.bzl) exports
[`create_et_provider()`](ETProvider.h). Applications must also link the kernel
libraries for their routes; the default compositions use
`optimized_native_cpu_ops`. The `provider_test` and `plan_test` targets cover
execution and integration with the planner.

See the [CPU README](../../../README.md) for provider composition and preferences.
