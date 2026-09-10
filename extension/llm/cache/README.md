# Off-graph KV cache

A KV cache that lives outside the exported graph. The runner creates it, the
backend writes into it during a forward, and neither holds a pointer to the
other. They meet through a string key.

Keeping the cache out of the graph lets the runner do things the graph cannot
express: rewind a turn, clear between prompts, or hand one pool of memory to
several concurrent sequences.

## Who uses what

Three audiences touch this directory, and they need almost disjoint parts of
it.

### The control plane

The runner, or a batch executor. It decides *what* to cache and *when* to
discard it, and it runs between forwards, never during one.

It uses `CacheFactory` to build a cache, `InstallGuard` to publish it, and one
runner-facing face for the rest of the session:

```cpp
auto built = CacheFactory::global().build(kMLXBackendId, kind::kSingle, cfg);
if (!built.ok()) { return built.error(); }

const std::shared_ptr<Cache> kv = built.get();
const InstallGuard guard{kv};                    // published while in scope
guard.set_option(mlx_opts);                      // hand key to backend

auto* ctl = kv->as<SequenceControl>();
ctl->can_extend(n);   ctl->rewind(len);   ctl->clear();
```

It includes `cache.h` and `cache_registry.h`. It never includes
`sequence_cache.h` or `cell_cache.h`, and never calls a planner face. It does
not know how bytes are arranged, only how much room is left and how to give
some back.

### The backend

The byte layer inside the delegate. It owns the actual tensors and runs during
a forward.

At init it resolves the key and asks for its own face:

```cpp
handle->cache_shared = CacheRegistry::global().get(cache_key);
handle->state.cache  = handle->cache_shared->as<MLXCache>();
```

During a forward it asks a planner face where the bytes go:

```cpp
auto plan = planner->plan(layer, position, T);   // integers, no tensors
// ... write K/V into those rows, attend over those runs ...
planner->commit(*plan);
```

It includes the layout headers, because it subclasses them to attach its own
tensor storage. It is the only caller of `plan()`, `commit()`, and `place_step()`.

### The cache implementer

Someone adding a layout or a backend face. Subclass a neutral layout, add
whatever face your backend needs, and list them:

```cpp
class MLXCellCache : public cache::CellCache, public MLXCache {
  void* face(cache::FaceId id) override {
    if (void* p = cache::CellCache::face(id)) { return p; }
    return cache::expose<MLXCache>(this, id);
  }
};
```

Then register a builder so the control plane can ask for it by name. Registration
is insertion-only: an empty builder or a duplicate `(backend_id, kind)` returns
`Error::InvalidArgument`, leaving any existing builder unchanged.

## Faces

A cache is owned as a `Cache*` and asked for the interface you want:

```cpp
auto* ctl = cache->as<SequenceControl>();   // null if this cache does not offer it
```

`Cache` has one virtual method. Each face declares its own name, and an
implementation lists the faces it offers through `expose`.

|                  | single sequence   | pooled cells   |
| ---------------- | ----------------- | -------------- |
| control plane    | `SequenceControl` | `BatchControl` |
| backend          | `SequencePlanner` | `CellStepper`  |
| backend-specific | each backend names its own, such as `MLXCache` ||

Control-plane faces live in `cache.h`. A runner calls `as<SequenceControl>()`
on something it got from the registry, so it must see the face without choosing
a layout. Backend faces live with their layout in `sequence_cache.h` or
`cell_cache.h`, because only a byte layer calls them and it already includes
that header to construct the cache.

**No RTTI.** The core avoids `dynamic_cast` so it can build with `-fno-rtti`
under `EXECUTORCH_OPTIMIZE_SIZE`. The `static_cast` inside `expose` also
applies the pointer adjustment a face at a non-zero offset needs, refuses to
compile if the type is not really a base, and is bound to its own name, so the
two cannot be mismatched. Because `as<T>()` names `T::kFaceName`, asking for a
type that is not a face fails to compile instead of returning null.

**Names, not an enum.** The set of faces is open. A backend adds one without
this directory learning about it: `MLXCache` declares its name in `MLXCache.h`
and `cache.h` never sees it. Names compare by pointer first and fall back to
`strcmp`, which covers a cache built in one shared object and queried from
another.

A face name is a global ABI identifier. It must be non-null, remain stable, and
identify exactly one C++ interface across the core, every backend, and every
shared object. Reusing a name for an unrelated or incompatible interface makes
the erased pointer cast invalid. The raw lookup hook is protected; consumers use
`as<T>()`, and each concrete cache must explicitly implement the faces it offers.

## Layouts

**`SequenceCache`** holds one sequence with a single logical length for the
whole model. Each layer is flat, keeping all history, or ring, sliding a
window, so a model that mixes both stays coherent. Offers `SequenceControl` and
`SequencePlanner`.

**`CellCache`** holds many sequences over a shared pool of per-token cells. A
cell is freed once no sequence owns it. Offers `BatchControl` and `CellStepper`.

## Cache kinds

| constant             | value           | layout                            |
| -------------------- | --------------- | --------------------------------- |
| `kind::kSingle`      | `single`        | one sequence, per-layer runs      |
| `kind::kBatchedCell` | `batched-cell`  | many sequences over a shared pool |

Kinds are strings so a backend can register a layout this directory has never
heard of. The constants name the kinds it does know about. Use them: a typo in
a literal is a runtime `NotFound`, while a typo in a constant does not compile.

## Lifetimes

`InstallGuard` is the only way to publish. `CacheRegistry::install` is private,
so an entry cannot outlive its owner and a second caller cannot clobber it.

Three lifetimes overlap:

- The **registry entry** must exist across every `load_method()` that resolves
  the key.
- The **guard** controls that discoverability and may be destroyed after the
  final such initialization.
- The **cache** may outlive the entry and guard. Each backend that resolved the
  key holds its own `shared_ptr`.

Destroying the guard unpublishes the key without invalidating an already
resolved cache. A later `load_method()` using that key fails, so the guard must
remain alive for as long as new delegates may still need to resolve it.

## Two layers

`cache.h`, `sequence_cache.h`, and `cell_cache.{h,cpp}` include nothing but the
C++ standard library. No tensors and no ExecuTorch. They describe where bytes
go using integers: which physical rows a step writes, which it reads, what the
mask should be. Failures come back as `bool` and `std::optional`.

`cache_registry.{h,cpp}` is ExecuTorch-specific. It uses `Result`, `Error`, and
`ET_LOG`, but the stronger tie is its reason for existing. `DelegateHandle` is
opaque and backend options carry only strings, so a runner cannot pass the
backend a pointer. Publishing under a generated key works around that. Give a
framework where the cache can be handed to the op directly, and this layer
disappears.

> The build does not yet honour this split. One `extension_llm_cache` target
> compiles both halves and links `executorch_core`, so the neutral core cannot
> currently be built without ExecuTorch.

## Files

```
cache.h                 faces, the face mechanism, config           neutral
sequence_cache.h        SequenceCache, SequencePlanner, flat/ring    neutral
cell_cache.{h,cpp}      CellCache, CellStepper, the cell pool        neutral
cache_registry.{h,cpp}  CacheRegistry, CacheFactory, InstallGuard    ExecuTorch
```

## Known gaps

**`CacheConfig` fields do not mean the same thing to every layout.** `capacity`
is a position ceiling for `kSingle` and a count of pool slots for
`kBatchedCell`. `max_write` is read only by ring layers, which in turn ignore
`initial_capacity`. Splitting it into model-dictated shape and per-kind options
is the intended fix.

**Registration relies on static-initializer side effects.** A builder is
registered only if the linker pulls its object file in. Whole-archive linking
of backends covers this today. If that changes, the symptom is a runtime
"no cache builder registered".
