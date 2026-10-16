# Native CPU engine

`CPUEngineHost` implements the Native engine API using the existing CPU
backend's `CPUPlan`, storage requirements, provider selection, and execution.
It follows the host/context/executable lifecycle of the Vulkan engine in
D119601866. It consumes `.ptn` packages directly through `ptn::Model` or the
lower-level `Program` and `Package` APIs.

Link `//executorch/backends/native/runtime/cpu:cpu_engine`. The default
composition includes XNNPACK, the Reference provider, and the existing optimized
Native CPU kernel registrations. A supplied `RuntimeConfiguration` can select
other linked provider factories and set preferences. The
factory list and preferences are copied when the host is created.

```cpp
auto host = ptn::CPUEngineHost::create();
auto model = ptn::Model::load_file("model.ptn");
auto session = model.create_session(*host);
session.prepare("forward");
session.run("forward", inputs, outputs);
```

The host must outlive its contexts/sessions, and a context must outlive its
executables. Calls through one context are serialized by the caller. Separate
contexts have independent activation arenas and input staging. Execution uses the shared ExecuTorch thread pool.

This version accepts static, unquantized tensor graphs with contiguous constants
and public I/O. It consumes the standard Native preprocessing output: at runtime
compilation, view/permute/transpose aliases and contiguous full-span as_strided
reshapes are materialized, and relu_/gelu_/sigmoid_ activations are functionalized.
Later reads of aliases with
the same shape and layout follow the updated value. Mutation through a live view
with different logical indexing is rejected, as are mutation of user inputs or
persistent buffers, control flow, dynamic shapes, and quantization metadata.
The linked providers determine which operators and dtypes can run. Unsupported
operators, including retained runtime assertions, fail provider selection.

Direct engine clients set every input before the first execution. Input values
remain available across repeated executions; setting an input invalidates
previous outputs. Host I/O copies use exact logical extents and accept unaligned
host pointers. Numeric dtype conversion preserves the logical graph dtype and
rejects out-of-range conversion to integers. Integer values are not narrowed
to int32 for execution.

Constants are read from the package into aligned, padded buffers. Compatible
requests for the same package owner, including package aliases, share a buffer
within the context. Those source buffers remain resident until context
destruction; providers may also allocate packed representations.
`unique_constant_bytes()` and `materialized_constant_bytes()` count logical
source bytes, excluding padding, activation arenas, and provider packing.

Run the integration tests with:

```sh
buck2 test @fbcode//mode/dev fbcode//executorch/backends/native/test/runtime/cpu:cpu_engine_test
```
