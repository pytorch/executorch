# XNNPACK provider

[`XNNPACKProvider`](XNNPACKProvider.cpp) compiles selected CPU regions into
XNNPACK subgraphs. It is the primary provider in the default CPU composition,
with one implementation, `XNNPACK/subgraph`. Each supported target is one `kOps`
entry that pairs its support check with its subgraph lowering.

The provider shares a workspace and weights cache across its regions. Compatible
convolution-weight conversions are reused within a plan; weights whose physical
order already matches are borrowed. Region boundaries use buffers managed by
[`CPUPlan`](../../CPUPlan.h).

The `cpu_xnnpack` target in [`targets.bzl`](../../../targets.bzl) exports
[`create_xnnpack_provider()`](XNNPACKProvider.h). Both `cpu_backend` and
`cpu_backend_dwconv7x7` link it. The `provider_test` and `plan_test` targets cover
execution and integration with the planner.

See the [CPU README](../../../README.md) for provider composition and preferences.
