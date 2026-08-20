# Direct native lowering (`to_native`)

`to_native` lowers exported PyTorch programs for the native portable runtime and
produces a `.ptn` package. Where the ExecuTorch flow is

```
export -> [quantize] -> to_edge -> to_backend -> to_executorch -> .pte (+ .ptd)
```

this flow is

```
export -> [quantize] -> to_native -> .ptn
```

## Usage

```python
import torch
from executorch.exir.native import to_native

ep = torch.export.export(model, example_inputs)
program = to_native(ep)
program.save("model.ptn")
```

Multiple methods are lowered together by passing a dict:

```python
program = to_native({"forward": ep_forward, "prefill": ep_prefill})
print(program.methods)          # {"forward", "prefill"}
program.save("model.ptn")
```

## API

### `to_native(programs) -> NativeProgramManager`

`programs` is a single `ExportedProgram`, lowered as the sole `forward` method, or
a non-empty dict mapping non-empty string method names to `ExportedProgram`.

Nothing about lowering is configurable yet, intentionally. We will open if
needed in the future like `constant_methods` and ETRecord integration.

Raises `TypeError` for the wrong input or method-value types. Raises `ValueError`
for an empty method mapping or method name, if a method does not fully delegate to
the native backend, if a constant differs between methods, or if a data key the
graph references has no backing tensor.

## Current limitations: one complete delegate per method

Each method must lower to **exactly one** native delegate that covers the whole
method, and the top-level graph must be an identity wrapper around it. This is not
"every `ExportedProgram` the native serializer can represent" — these cases are
outside the contract and fail loudly rather than producing a partial package:

- **Identity methods** (`return x`) delegate nothing, so there are zero delegates.
- **Disconnected supported regions** become several partitions.
- **Graph breaks** leave an unsupported op in the outer graph. In the ExecuTorch
  path that op would run outside the delegate; here there is no outer runtime.
- **Outer-graph plumbing** that drops, reorders, duplicates, or injects values —
  including a pass-through like `return x, x + 1` — changes the method's contract
  and cannot be represented by packaging the delegate alone.

Validation happens before a `NativeProgramManager` exists, and `save` is a separate
call, so a rejected program never writes a file.

TODO: compare the merged method's complete output arity, mutation targets and
ordering, and dynamic-shape constraints. The current wrapper check rejects outer-graph
plumbing it can observe, but these signature details need merged-program metadata.

### `NativeProgramManager`

The counterpart to `ExecutorchProgramManager`. `methods` and `save` mirror it
deliberately, so the two pipelines read alike.

| Member | Description |
| --- | --- |
| `methods` | Names of the methods in the program. |
| `save(path)` | Write the program and its constants to `path` as a `.ptn`. |

The surface is intentionally small. TODO: add `buffer`, `exported_program`,
`write_to_file`, and debug-map access only when concrete callers establish the
required contracts; these additions can remain additive.

Constants are held as tensor references returned by lowering. `save` may normalize
device or layout before writing them.

## The `.ptn` package

An uncompressed zip (TODO: revisit the packaging format later) with fixed member names:

| Entry | Contents |
| --- | --- |
| `program.ptg` | Native Program flatbuffer, file identifier `NPTG`. |
| `program.safetensors` | Constants the graph references; immutable values may be content-deduped. Present only when the graph references constants. (TODO: revisit this format later) |
| `aliases.json` | Duplicate data key to owner key. Present only when duplicates exist. |

Byte-identical immutable tensors are stored once: the first key owns the
`safetensors` entry and the rest alias to it. An owner is always a real
`safetensors` key and never appears in the alias map, so resolution is
`owner = aliases.get(key, key)`.

`save` writes each `safetensors` owner payload straight into the zip instead of
materializing the whole model as Python bytes. Owner tensors remain live until the
write completes, and non-CPU or non-contiguous inputs may require normalized tensor
copies.

Only **immutable** constants are deduplicated. An alias means "these keys held
identical bytes at save time, so one copy was stored" — never that two keys share
runtime state. Mutable buffers are excluded, because two independently mutable
buffers can be zero-initialized to identical bytes and must not end up aliased.
Distinct data keys that share source storage are rejected when either is mutable:
The current PTN format cannot encode that alias topology, and storing the keys
independently would silently change mutation semantics. TODO: add explicit storage
groups and view metadata if distinct-key mutable aliasing becomes necessary.

TODO: save atomically — write beside the destination and replace it once complete,
so a failure cannot truncate an existing valid package.

## Relationship to the ExecuTorch path

`NativePartitioner` remains usable directly through
`to_edge_transform_and_lower`, in which case lowering behaves like any other
ExecuTorch delegate: the graph is a delegate blob and constants are shipped via the
`NamedDataStore`, honoring `external_constants_tag`. That path is unchanged and
still produces a `.pte`.

`to_native` opts the partitioner into handing constants back out of band instead,
which is why it returns a terminal `NativeProgramManager` rather than an
`EdgeProgramManager`: such a program has no constants to serialize into a `.pte`.

## Layout

| Path | Contents |
| --- | --- |
| `exir/native/` | This API: `to_native`, `NativeProgramManager`. |
| `exir/_serialize/_ptn.py` | The `.ptn` format, independent of any backend. |
| `backends/native/` | The backend: partitioner, preprocess, passes, and the `NPTG` graph serializer. |
