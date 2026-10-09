# CPU Backend

> [!NOTE]
> Do not use.

The initial workload is pretrained ConvNeXt-Tiny in FP32 on a Linux server.
The registered `CPURecipeType.FP32` (`cpu_fp32`) Recipe retains a semantic Native
graph inside one `CpuBackend` delegate in a regular ExecuTorch `.pte`.
This first delegate version supports pure operations on statically shaped,
contiguous FP32 tensors.

The `cpu_delegate_version` compile spec stores `CPU_DELEGATE_VERSION` as a
four-byte little-endian unsigned integer. The loader accepts explicitly
supported versions. Bump the version when changing serialized semantics or
storage guarantees incompatibly; provider additions, kernel optimizations, and
runtime planning changes do not require a bump. Native graph schema versioning
is separate.


The C++ runtime selects implementations from linked `KernelProvider` factories.
XNNPACK is the linked provider; other providers can supply fallback when no
XNNPACK implementation accepts an operation. Selection precedes XNNPACK region
compilation, so compiled regions execute within one CPU schedule and share
boundary buffers. Provider choices are absent from
the serialized graph.

Provider implementations live under `runtime/providers/`:

- [`xnnpack/`](runtime/providers/xnnpack/README.md): XNNPACK subgraph compilation.

The shared `KernelProvider` contract, `CPUPlan`, and `CpuBackend` remain in
`runtime/`. Provider targets and application compositions are defined in
[`targets.bzl`](targets.bzl).

The Buck `cpu_backend` target links the XNNPACK provider.
Applications compose providers with [`cpu_backend()`](cpu_backend.bzl).

`CPUPlan` retains selection, regions, storage requirements and prepared executables
separately from the loader's Native graph and current buffer bindings. Providers
receive Native metadata and explicit buffer extents. Requirements are queried
before constant preparation and arena
allocation. Sequential steps share the maximum declared scratch capacity. XNNPACK
shares compatible convolution-weight conversions within a plan and borrows weights
whose physical order already matches.

For a loaded method, shapes, FP32 dtype, contiguous layout, threadpool identity,
thread count and ISA capabilities remain fixed. Input/output pointers may change
and are validated on every invocation. Changing an execution property requires
reloading the method; failed validation executes no provider. The trace includes
candidate eligibility/rejection reasons and the captured execution properties.
Priority selection still has unknown cost. Dynamic shape inference, transactional
plan replacement, resource bindings and memory budgets remain future work.

XNNPACK workspace and packed-cache capacities remain unknown. These are separate
from the reported boundary arena, shared scratch and converted constants.

Applications specify an ordered list of `(provider, implementation)` preferences
in `cpu_backend(preferences=[...])`. An empty implementation matches any
implementation from that provider. For example, this hypothetical composition
would prefer two specific kernels, then YNNPACK for the remaining operations:

```python
preferences = [
    ("XNNPACK", "DWCONV5x5"),
    ("CUSTOM", "DWCONV7x7"),
    ("YNNPACK", ""),
]
```

These names illustrate the policy; YNNPACK and the named 5x5 implementation are
not supplied by this prototype. Every configured provider and implementation must
exist in the linked composition. For each operation, the first matching eligible
entry wins. Numeric baseline priority resolves candidates matching the same
entry, and supplies fallback if no preference applies. An exact entry does not
prefer other implementations from that provider. Preferences never bypass
eligibility or storage checks. `force=True` requires at least one operation to
select a preferred implementation.

The following diagrams use `*` for a provider-wide entry (an empty implementation
string in the API). Selection is evaluated separately for each operation:

```text
                  operation
                      |
                      v
 [0] XNNPACK / DWCONV5x5 ---- eligible ----> select XNNPACK / DWCONV5x5
                      |
                  ineligible
                      |
                      v
 [1] CUSTOM / DWCONV7x7 ----- eligible ----> select CUSTOM / DWCONV7x7
                      |
                  ineligible
                      |
                      v
 [2] YNNPACK / * ----------- eligible ----> select highest-priority
                      |                    eligible YNNPACK implementation
            no eligible implementation
                      |
                      v
       normal baseline priority among all
           linked eligible implementations
```

Order also matters when both a specific kernel and the later provider can handle
the same operation. For a 7x7 convolution supported by both CUSTOM and YNNPACK:

```text
 Specific kernel first                  Provider first
 ---------------------                  --------------
 [0] CUSTOM / DWCONV7x7                  [0] YNNPACK / *
 [1] YNNPACK / *                         [1] CUSTOM / DWCONV7x7
           |                                      |
           v                                      v
 select CUSTOM / DWCONV7x7               select an eligible YNNPACK kernel
```

The earlier entry wins even if the later candidate has a higher numeric baseline
priority. If the earlier entry is ineligible for this operation, selection moves
to the next eligible entry.

Per-model `CpuBackend` load options can replace the entire list: set integer
`preference_count`, then string `preferred_provider_0`,
`preferred_implementation_0`, and so on in preference order. Implementation keys
are optional; a count of zero clears the list. For example, these options prefer
the XNNPACK subgraph implementation:

```cpp
BackendOptions<4> options;
options.set_option("preference_count", 1);
options.set_option("preferred_provider_0", "XNNPACK");
options.set_option("preferred_implementation_0", "subgraph");
```

Check each `set_option` result, attach the options under `"CpuBackend"` in a
`LoadBackendOptionsMap`, and pass it to `Module::load()`. Strings are copied into
the plan. The existing unindexed `preferred_provider` and
`preferred_implementation` options remain a single-pair shorthand and cannot be
combined with `preference_count`. They update the existing pair when there is
exactly one; otherwise they replace the list. The Boolean `force` option is
independent and retains its build default when omitted.

The test runner accepts up to eight `provider[/implementation]` arguments after
the model and golden paths, in preference order.
