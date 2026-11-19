# XNNPACK provider

[`XNNPACKProvider`](XNNPACKProvider.cpp) compiles selected CPU regions into
XNNPACK subgraphs. It is the primary provider in the default CPU composition,
with one implementation, `XNNPACK/subgraph`. Each supported target is one `kOps`
entry that pairs its support check with its subgraph lowering.

FP32 `avg_pool2d` accepts static rank-four NCHW tensors, floor or ceil mode,
and either `count_include_pad` setting. Global pooling uses a mean reduction;
other windows use NHWC average pooling, with declared padding materialized when
it must count toward the divisor. Explicit divisor overrides and kernels larger
than the padded input remain on Reference, as do unsupported tensor metadata
and dynamic arguments.

Before lowering a region, [`XNNPACKPasses`](XNNPACKPasses.h) recomposes ATen's
decomposed `hardswish` into one `xnn_unary_hardswish` node when its
intermediates stay inside the region.

The provider shares a workspace and weights cache across its regions. Compatible
convolution-weight conversions are reused within a plan. They and broadcast
linear biases are freed once the weights cache is finalized; weights whose
physical order already matches are borrowed. Region boundaries use buffers managed by
[`CPUPlan`](../../CPUPlan.h).

The `cpu_xnnpack` target in [`targets.bzl`](../../../targets.bzl) exports
[`create_xnnpack_provider()`](XNNPACKProvider.h). Both `cpu_backend` and
`cpu_backend_dwconv7x7` link it. The `provider_test` and `plan_test` targets cover
execution and integration with the planner.

See the [CPU README](../../../README.md) for provider composition and preferences.
