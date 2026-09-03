# Off-graph KV cache

A KV cache that lives outside the exported graph. The runner creates it, the
delegate writes into it during a forward, and neither holds a pointer to the
other — they meet through a string key.

Exporting the cache out of the graph lets the runner do things the graph can't
express: rewind a turn, clear between prompts, or hand one pool to several
concurrent sequences.

## Two layers

**The core is framework-neutral.** `cache.h`, `sequence_cache.h`, and
`cell_cache.{h,cpp}` include nothing but the C++ standard library. No tensors,
no ExecuTorch. They describe *where bytes go* in integers — which physical rows
a step writes, which it reads, what the mask should be — and hand that to
whoever owns the actual memory. Failures come back as `bool` and
`std::optional`.

**The registry is ExecuTorch-specific.** `cache_registry.{h,cpp}` uses
`Result`, `Error`, and `ET_LOG`, but the deeper tie is its reason for existing:
`DelegateHandle` is opaque and backend options carry only strings, so a runner
cannot pass the delegate a pointer. Publishing under a generated key and
meeting on that string is a workaround for that constraint. In a framework
where you can hand the cache to the op directly, this layer disappears.

> The build does not yet honour this split: one `extension_llm_cache` target
> compiles both halves and links `executorch_core`, so the neutral core cannot
> currently be built neutrally.

## Faces

A cache is owned as a `Cache*` and asked for the interface you want:

```cpp
auto* ctl = cache->as<SequenceControl>();   // null if not offered
```

`Cache` has one virtual. A face declares its own name, and an implementation
lists the ones it offers:

```cpp
class MLXCache {
  static constexpr const char* kFaceName = "mlx.MLXCache";
};

void* face(FaceId id) override {
  return expose<BatchControl, CellStepper>(this, id);
}
```

**Why not `dynamic_cast`.** The neutral core avoids RTTI so it can be built
with `-fno-rtti` under `EXECUTORCH_OPTIMIZE_SIZE`. `static_cast` inside
`expose` also applies the pointer adjustment a face at a non-zero offset needs,
refuses to compile if the type is not really a base, and is bound to its own
name so the two cannot be mismatched. Naming `T::kFaceName` makes
`as<NotAFace>()` a compile error rather than a silent null.

**Why names rather than an enum.** The set is open. A backend adds a face
without this directory learning about it — `MLXCache` is declared in
`MLXCache.h` and `cache.h` never sees it. Names are compared by pointer first,
falling back to `strcmp` only when a cache crosses a shared-object boundary.

### The faces

|                  | single sequence   | pooled cells   |
| ---------------- | ----------------- | -------------- |
| runner-facing    | `SequenceControl` | `BatchControl` |
| backend-facing   | `SequencePlanner` | `CellStepper`  |
| backend-specific | each backend names its own (`MLXCache`)  ||

Runner-facing faces live in `cache.h`: a caller does `as<SequenceControl>()` on
something it got from the registry and must see the face without choosing a
layout. Backend-facing faces live with their layout, because only a byte layer
calls `plan()` and it already includes that header to construct the cache.

## Layouts

**`SequenceCache`** — one sequence, one logical length for the whole model,
per-layer runs. Each layer is flat (keeps everything) or ring (slides a
window), so a mixed model stays coherent. Offers `SequenceControl` and
`SequencePlanner`.

**`CellCache`** — many sequences sharing a pool of per-token cells. A cell is
freed when no sequence owns it. Offers `BatchControl` and `CellStepper`.

A backend subclasses one and adds its own face:

```cpp
class MLXCellCache : public cache::CellCache, public MLXCache {
  void* face(cache::FaceId id) override {
    if (void* p = cache::CellCache::face(id)) { return p; }
    return cache::expose<MLXCache>(this, id);
  }
};
```

## The rendezvous

**1. Register**, once, in a static initializer in the backend:

```cpp
CacheFactory::global().register_builder(
    kMLXBackendId, cache::kind::kSingle,
    [](const cache::CacheConfig& cfg) {
      return std::shared_ptr<cache::Cache>(std::make_shared<MLXSequenceCache>(cfg));
    });
```

**2. Build.** The factory validates the config, rejects a null builder result,
and on an unknown kind names the kinds that do exist.

```cpp
auto built = CacheFactory::global().build(kMLXBackendId, cache::kind::kSingle, cfg);
if (!built.ok()) { return built.error(); }
```

**3. Publish.** `InstallGuard` mints a key, installs the cache, and erases the
entry when it goes out of scope. It is the only way to publish —
`CacheRegistry::install` is private — so an entry cannot outlive its owner or
be clobbered by a second caller.

```cpp
const cache::InstallGuard guard{built.get()};
mlx_opts.set_option(kCacheKeyKey, guard.key());
```

**4. Resolve.** Backend init runs inside `load_method()` and is the only place
the registry is read:

```cpp
handle->cache_shared = CacheRegistry::global().get(cache_key);
handle->state.cache  = handle->cache_shared->as<MLXCache>();
```

**5. Teardown.** `~InstallGuard` erases the entry. The delegate still holds its
own `shared_ptr`, so the cache survives — only findability by key ends.

### Three lifetimes

- The **entry** must exist across `load_method()`, which is when init resolves
  the key. Nothing reads the registry after that.
- The **guard** usually lives longer, because the runner reaches the cache
  through its own pointer for `can_extend` and `clear`.
- The **cache** outlives both under shared ownership.

Destroying the guard early does not dangle anything. It silently unpublishes,
and the next `load_method()` fails with a key that is no longer installed —
a delayed, confusing failure rather than a crash. Nothing enforces the
ordering; keep the guard in a scope that outlives the module.

## Cache kinds

| constant             | value           | layout                             |
| -------------------- | --------------- | ---------------------------------- |
| `kind::kSingle`      | `single`        | one sequence, per-layer runs       |
| `kind::kBatchedCell` | `batched-cell`  | many sequences over a shared pool  |

Kinds are strings so a backend can register a layout this directory has never
heard of. The constants are the vocabulary for the ones it knows about, not an
enumeration of what is valid — go through them, because a literal typo is a
runtime `NotFound` rather than a compile error.

## Files

```
cache.h              faces, the face mechanism, config              neutral
sequence_cache.h     SequenceCache + SequencePlanner, flat/ring      neutral
cell_cache.{h,cpp}   CellCache + CellStepper, the cell pool          neutral
cache_registry.{h,cpp}  CacheRegistry, CacheFactory, InstallGuard    ET
```

## Known gaps

**`CacheConfig` fields are not read the same way by every layout.** `capacity`
is a position ceiling for `kSingle` but a count of pool slots for
`kBatchedCell`; `max_write` is read only by ring layers, which in turn ignore
`initial_capacity`. Splitting it into model-dictated shape and per-kind options
is the intended fix.

**Registration relies on static-initializer side effects**, so a builder is
registered only if the linker pulls its object file in. That works today
through whole-archive linking of backends; if it ever stops, the symptom is a
runtime "no cache builder registered".
