# Native CPU engine

`CPUEngineHost` implements the Native engine API using the existing CPU
backend's `CPUPlan`, storage requirements, provider selection, and execution.
It follows the host/context/executable lifecycle of the Vulkan engine in
`runtime/vulkan/`. It consumes `.ptn` packages directly through `ptn::Model` or
the lower-level `Program` and `Package` APIs.

Link `//executorch/backends/native/runtime/cpu:cpu_engine`. The default
composition includes XNNPACK, the Reference provider, and the existing optimized
Native CPU kernel registrations. A supplied `RuntimeConfiguration` can select
other linked provider factories and set preferences; an empty provider list
selects XNNPACK and Reference. The factory list and preferences are copied when
the host is created, so their source storage need not outlive `create()`.

```cpp
auto host = ptn::CPUEngineHost::create();
auto model = ptn::Model::load_file("model.ptn");
auto session = model.create_session(*host);
session.prepare("forward");
session.run("forward", inputs, outputs);
```

The host must outlive its contexts/sessions, and a context must outlive its
executables. Calls through one context are serialized by the caller. Separate
contexts have independent activation arenas and input buffers. Execution uses
the shared ExecuTorch thread pool, read on every execution; after the pool is
reset, executions fail until the method is compiled again.

The engine is a work in progress. The goal is to consume the standard Native
preprocessing output; this version accepts a subset of it: static, unquantized
tensor graphs with contiguous constants and public I/O. At runtime compilation,
the shared `functionalize_aliases` pass (`runtime/graph/utils/Aliases.h`)
materializes view/permute/transpose aliases and contiguous full-span as_strided
reshapes, and functionalizes relu_/gelu_/sigmoid_ activations. Other
aliasing views (for example slice, unsqueeze and expand) and in-place
index_put_/index_copy_ are rejected. Later reads of aliases that reach the
mutated value through views alone, with the same shape, follow the updated
value. Mutation through a live view
with different logical indexing is rejected, as are mutation of user inputs or
persistent buffers, control flow, dynamic shapes, and quantization metadata.
After this pass, intermediates that are not graph outputs are stored
contiguously regardless of their dim-order hint; every remaining value,
including graph inputs and outputs, must be a contiguous tensor of rank at most
16. The linked providers determine which operators and dtypes can run.
Unsupported operators, including retained runtime assertions, fail provider
selection.

Direct engine clients set every input before the first execution. Inputs are
written once into dedicated buffers outside the activation arena, so their
values remain available across repeated executions; setting an input
invalidates previous outputs. Host I/O copies use exact logical extents and
accept unaligned host pointers. Numeric dtype conversion preserves the logical
graph dtype. Conversion to integers truncates toward zero, as in C++, and
rejects values whose truncated result is out of range. Integer values are not
narrowed to int32 for execution.

Constants are read from the package into aligned, padded buffers. Compatible
requests for the same package owner, including package aliases, share a buffer
within the context, across methods. Those source buffers remain resident until
context destruction; providers may also allocate packed representations.
Constants loaded by a method compilation that fails are released, leaving the
context unchanged. `unique_constant_bytes()` and
`materialized_constant_bytes()` count logical source bytes, excluding padding,
activation arenas, and provider packing.

Run the integration tests with:

```sh
buck2 test @fbcode//mode/dev fbcode//executorch/backends/native/test/runtime/cpu:cpu_engine_test
```
