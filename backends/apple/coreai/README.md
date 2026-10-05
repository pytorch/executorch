# Core AI backend (ExecuTorch)

> ⚠️ **Under construction — not for use.**
> This backend is a work in progress. It is **not** ready for production or
> general use, and provides **no backward- or forward-compatibility guarantees**
> (no BC/FC). APIs, compile specs, serialized manifests, and file layouts may
> change or break at any time without notice. Do not depend on it.

## Running the tests

Install ExecuTorch in editable mode, then the pinned Core AI SDK packages:

```bash
python install_executorch.py --editable
pip install -r backends/apple/coreai/requirements.txt
```

Run the backend's Python tests from the repository root:

```bash
python -m pytest backends/apple/coreai
```

`CoreAIAOTCompileTest` needs `xcrun coreai-build` from the Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`) and skips without it.

## Runtime

The delegate manifest describes `.aimodel` source bundles, which require
`inline` packaging, and AOT-compiled `.aimodelc` bundles, which require
`aot_compiled_inline`. Both formats require a `function` name, and ordered
`input_names` and `output_names` matching the converter's bindings. Dictionary
iteration order is not a binding contract. The `files` object maps relative
filenames to byte sizes; `bundle_digests` maps each bundle basename to its
export-time SHA-256 digest.

The experimental `CoreAIBackend` runtime supports stateless FP32/FP16 tensor
models in both formats.
ExecuTorch tensors must be contiguous. Output shapes
must fit the executor's declared resize and capacity constraints. Stateful
functions, image values, other dtypes, and interleaved layouts are not supported.
Calls on one session must be serialized by the caller.

Backend `init` and `execute` block the calling thread until Core AI finishes,
and that work runs on Swift's shared concurrency pool. Do not load or execute
from Swift concurrency code (an `async` function or `Task`), because blocking
those threads can starve the pool; call from a dedicated thread or dispatch
queue instead.

Supported inputs are borrowed directly from ExecuTorch storage through Core AI
raw views. The bridge copies shape metadata, not tensor bytes, and does not
allocate intermediate input `NSData`, `Data`, or `NDArray` buffers. Each call
wraps `const_data_ptr()` and the current shape's `nbytes()`, not the allocation's
maximum capacity. Outputs use the SDK's actual returned shape; the backend
resizes them within ExecuTorch's declared bounds before copying only the logical
result bytes. Unused capacity is neither wrapped as input nor written as output.
The backend waits for inference and all input access to finish before returning
from `execute`, including on errors. This uses ExecuTorch's normal input-lifetime
contract; callers must not concurrently mutate, release, or rebind the input
storage. Output data remains owned and is copied into validated ExecuTorch
outputs after inference, so output/input aliases are not written during the
borrow. This does not guarantee that Core AI itself avoids device transfers or
internal copies.

### AOT architecture selection

Configure AOT export with `AOTCompileConfig`, including the target `platform` and
optional `architectures` list. For example, `AOTCompileConfig(platform="iOS",
architectures=["h17p"])` requests that compiler architecture; omitting the list
lets the compiler emit its supported architectures for the target platform.
These are Core AI architecture names, not CPU names such as `arm64`.

At load time, the runtime queries `AIModel.deviceArchitectureName` and selects
the exact matching entry in the manifest's `archs` map. No runtime architecture
option is required, and there is no fallback to another architecture. Platform,
deployment floor, and architecture checks happen before asset reads or storage
creation. Missing-architecture errors identify the device architecture, list
available architectures, and request a matching export. Missing selected files
are reported separately from an absent architecture entry.

Both formats can require device specialization. Every load first attempts SDK
bookmark restoration. A missing bookmark or SDK-confirmed cache miss falls back
to source materialization and persistent specialization, then binds the requested
function and ordered I/O on that same acquired model.

### Asset storage

Backend initialization reads the string runtime spec `coreai_assets_dir`. If
absent, it sets the directory to `executorch_coreai` beneath user-domain
`NSCachesDirectory`. The coordinator and storage helpers always receive that
explicit directory. It contains raw bookmarks, staged bundles and lock files.

The assets root must be an absolute path. The application chooses it, should
reserve it for this backend, and is trusted not to rename, replace or modify its
contents while the backend uses it; the backend does not defend against other
processes of the same user rebinding paths. Written files and changed directories
are synced with `fsync`; atomic publication also uses `F_FULLFSYNC` before its
rename.

Inline bundles are read through `NamedDataMap` and materialized as unchanged
files under the assets root. Named data can be supplied externally; inline
packaging does not require all bytes to reside in one physical PTE. The exporter
computes each bundle digest from all delivered relative filenames and file bytes,
not just the SDK's bytecode-only `main.hash`. Each AOT architecture has an
independent digest; only the selected architecture is materialized. NDS key names
and raw asset bytes are unchanged.

When source recovery is needed, existing bundles are checked for the expected
file set, sizes, entry types and directory structure without reading asset
contents or requesting NDS payloads.
Missing bundles are written to a staging directory and published with an
exclusive rename; each selected NDS payload is fetched once and checked for size
before writing. If another loader publishes first, its bundle is validated and
used. Incomplete, size-mismatched or otherwise malformed existing bundles fail
loading without replacement, since an SDK model may still be using them.

These are completeness checks, not full content-integrity verification. Same-size
content changes are not detected, and a complete stored bundle does not cause its
NDS source to be reread. The export-time digest is a trusted artifact identifier;
the runtime does not rehash source or stored bytes. Core AI may reject invalid
contents, but SDK load failures do not automatically evict the extracted copy.

Preparing the assets root sets `NSURLIsExcludedFromBackupKey` on it, which covers
everything beneath it. Ancestors are not modified. This is backup exclusion, not
a control for iCloud Drive synchronization.

Delegate construction and registration do not create storage directories or load
models. Per-model initialization performs SDK acquisition and function loading,
validating source assets only when recovery requires them. There is no synthetic
zero-input inference at initialization. SDK specialization and function resource
loading can still be substantial.

The runtime always uses `AIModelCache.default` with persistent retention and
stores an opaque bookmark for direct restoration. `coreai_assets_dir` does not
choose the SDK artifact directory. The backend does not inspect SDK-private
files, alter their backup flags, or use an App Group cache.

Each partition derives a key from its selected export digest/bundle, platform,
Core AI device architecture, default SDK cache namespace, versioned default
options and persistent policy. Function bindings and delivery location are
excluded. No model bytes are hashed at runtime. Each key determines a raw
bookmark in the `bookmarks` subdirectory, a bundle directory in `staging`, and an
empty coordination file in `locks`. There is no PTE-wide bookmark list or
serialized lifecycle record.

A healthy bookmark hit does not inspect staging or consult NDS; the acquired SDK
model is retained through function binding. On a missing bookmark or confirmed
SDK miss, the runtime materializes the source, specializes it and atomically
writes the returned bookmark bytes. There is no secondary source-URL cache
lookup. SDK errors, unreadable bookmarks and publication failures fail loading
without automatic cache deletion. If publication fails after specialization, the
SDK entry and source remain; the next load retries the ordinary flow. The
backend does not track historical or unrecorded SDK entries.

Raw bookmarks have an 8 MiB backend allocation limit, not an asserted SDK format
maximum, and are published atomically because losing one orphans an SDK entry.
Per-key disk locks coordinate loading and eviction across threads and processes;
different keys remain independent. Lock ownership is maintained by the OS, and
acquiring a lock does not flush its file or directory. Lock files are not
removed by the backend, including after process exit. Removing the assets root
externally requires all loads, sessions and maintenance to stop.

Automatic source removal is disabled. A durable bookmark and a live model pin
do not prove source independence. The backend does not remove sources on method
unload or delegate destruction; no runtime option enables automatic removal.
Cache storage and SDK persistent entries may be purged, so retain or redeliver
the original PTE and named data for reconstruction after a cache miss.
Source-free inference, fresh-process SDK restoration and persistent-policy
behavior require separate OS 27 hardware qualification.

### Clearing cached assets

The public C++ maintenance APIs are in `executorch::backends::coreai`:

```cpp
runtime::Error clear_cache(const char* coreai_assets_dir);
runtime::Error clear_cache_for_pte(
    runtime::DataLoader& pte, const char* coreai_assets_dir = nullptr);
runtime::Error clear_cache_for_pte(
    const char* pte_path, const char* coreai_assets_dir = nullptr);
```

`clear_cache` requires an explicit absolute assets root. The PTE overloads use
user-domain `NSCachesDirectory` plus `executorch_coreai` when the directory is
omitted or null, just like backend initialization. An invalid supplied path is
an error. If loading used a custom root, pass that same root when clearing;
the PTE does not store runtime directory overrides.

Root-wide clearing visits keyed bookmarks and staged directories. PTE clearing
selects and deduplicates keys for all Core AI delegates in every method, using
the current platform, SDK architecture and default specialization configuration.
Both public `clear_cache_for_pte` overloads use the private `inspect_coreai_pte`
reader and retain its `CoreAIPteData`, including the verified program image and
selected processed buffers, through `clear_keys`. Discovery does not load a
Program or initialize a Method, load SDK models, bind functions, extract Core AI
assets or specialize. All selected manifests are validated before any key is
cleared; root-wide clearing remains
available for existing keyed entries. Backend-local structural and semantic
verification is mandatory even with `ET_ENABLE_PROGRAM_VERIFICATION=0`.
The reader's I/O and buffer-lifetime contract is described below.

These synchronous calls require callers to unload affected models and prevent
concurrent loads, inference and maintenance across processes. Per-key locks are
not a whole-root barrier. Clearing first deletes the recorded SDK entry, then
staging, then its bookmark. Busy or unknown SDK errors preserve the files;
SDK-confirmed absence permits cleanup. Filesystem errors may leave partial
cleanup, but the bookmark is retained until staging removal succeeds so cleanup
can be retried. Independent keys are still attempted in sorted order, and the
first key error is returned. Each removal syncs the directory it changed. A
missing root or absent entry is a noncreating no-op.

Everything under a keyed `staging/<key>` directory belongs to the backend and is
removed with that key. Outside keyed entries, clearing does not delete PTEs,
other files or the stable lock files. Unkeyed interrupted-staging directories
also remain untouched. Bookmarks are SDK eviction handles, not weights: lost
bookmarks can leave SDK entries that these APIs cannot target. There is no
history or ownership registry. Copied or reused exported artifacts can share
keys; clearing them also removes the other copy's warm-cache benefit. Subsequent
loads can specialize again.

### Runtime options

Pass options through the public `Module` API before loading. In an error-returning
loader using the `executorch::runtime` namespace:

```cpp
BackendOptions<1> options;
ET_CHECK_OK_OR_RETURN_ERROR(options.set_option("coreai_assets_dir", assets_dir));
LoadBackendOptionsMap map;
ET_CHECK_OK_OR_RETURN_ERROR(map.set_options("CoreAIBackend", options.view()));
ET_CHECK_OK_OR_RETURN_ERROR(module.load(map));
```

No options are required when using the default assets root. `coreai_assets_dir`
selects an application-chosen cache directory reserved for this backend. The
supplied path is validated during preflight, but warm bookmark hits need no
source.

Use separate option maps to choose different storage roots for different models.
The path is a root directory, not the bundle itself. ExecuTorch option strings
have a 255-byte limit.

## SDK Bridge

The private Swift module `CoreAIBridge` (`runtime/ETCoreAIModel.swift`) wraps
the `CoreAI` framework behind the Objective-C protocols in
`runtime/ETCoreAIBridge.h`. It restores models from bookmarks, specializes source
bundles with the persistent default SDK cache and binds named functions with
ordered inputs and outputs. Asynchronous SDK calls run in detached tasks and
report through completion blocks; function binding completes synchronously.

Inputs are borrowed from caller storage through Core AI raw views; only shape
metadata is copied. Dense row-major outputs are copied with one `memcpy`, and
padded or transposed outputs fall back to strided copies. Only FP16 and FP32
NDArrays are supported.

The Objective-C half of the bridge builds as the private `coreai_bridge_obj`
object library, which production delegate targets absorb. Host tests substitute
a fake bridge instead.

`coreai_swift_bridge_test` runs the bridge's XCTests under `xctest` with
`EXECUTORCH_BUILD_TESTS=ON`. It links CoreAI, so it runs only on macOS 27.

## Building

Core AI builds require an Apple SDK 27 or newer, Swift from Xcode 27, and CMake
3.31 or newer with the Ninja or Xcode generator. Execution requires macOS 27 or
iOS 27. `EXECUTORCH_BUILD_COREAI` defaults to `OFF`. Enabling it builds the
private Swift implementation, module `CoreAIBridge`. Only Core AI targets get an
OS 27 minimum; unrelated runtime targets keep the configured deployment target.
Ninja builds use one architecture per build directory. Current runtime support
covers arm64 macOS and iOS device builds. x86_64 builds are blocked by Swift
`Float16` availability, and the tested iOS simulator SDKs do not contain Core AI.

Enable `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER` for inference consumers and
the public PTE-file maintenance API.
Link the CMake target `coreaidelegate`, not just its archive filename. Its
transitive dependencies supply Swift and Foundation/CoreAI linkage, and its
link interface retains static backend registration. With
`EXECUTORCH_BUILD_SHARED=ON`, the delegate is a shared library linked to the
single consolidated `executorch_shared` runtime. Otherwise it is static.
Consumers of the shared delegate must use that same shared runtime.
Installed CMake targets resolve framework and Swift runtime dependencies using
the consumer's selected SDK and toolchain, not the producer's Xcode paths.
Select the Apple SDK/toolchain when configuring the consumer project.
SwiftPM and XCFramework distribution are not integrated yet.

The `coreai_runtime_smoke` target checks final linkage of a C++ consumer,
including Swift dependencies, backend registration retention and all public
cache-clearing overloads. With `EXECUTORCH_BUILD_TESTS=ON` it is built and
registered with CTest; otherwise it is excluded from the default build. On OS 27,
running it checks registration and availability; it does not perform model
inference. Do not execute SDK27 binaries on an older host.

## Backend-local PTE Inspection

The private `inspect_coreai_pte(DataLoader&)` reader returns Core AI processed
buffers in method/delegate order without loading a Program or Method, registering
backends, or calling the SDK. It always verifies the program structure, supported
schema version, and existing runtime semantic invariants, even when generic
runtime verification is disabled (`ET_ENABLE_PROGRAM_VERIFICATION=0`). It also
validates method metadata, unique method names, and selected processed-data
references.

The reader first reads a 64-byte prefix when the input is long enough, then
loads and verifies the program region, including inline constants and unrelated
inline blobs. Only external processed-data segments selected by exact
`CoreAIBackend` IDs are requested; unselected external payloads, the external
constants segment and external named assets are not loaded. Inline buffers
borrow the retained program image. Selected segment buffers retain their
original DataLoader callbacks. Keep the returned `CoreAIPteData` owner and loader
alive while using the buffers, and keep the loader's data stable throughout
inspection and use. The reader does not derive cache keys or mutate cache state.

## Host Tests

`EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON` registers the
standalone `coreai_host_test` executable with CTest. It compiles the runtime
sources without Swift or CoreAI, routing any SDK calls through a fake bridge, so
it also runs on macOS 26. Host tests do not establish real SDK behavior.

From the repository root:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DEXECUTORCH_BUILD_COREAI=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
cmake --build <build-dir> --target coreai_host_test
ctest --test-dir <build-dir> -R '^coreai_host_test$' --output-on-failure
```

For a compile-only production smoke build, use a fresh build directory without
`EXECUTORCH_BUILD_TESTS`, set `CMAKE_OSX_ARCHITECTURES=arm64`, and build
`coreai_runtime_smoke`. For iOS also set `CMAKE_SYSTEM_NAME=iOS` and
`CMAKE_OSX_SYSROOT=iphoneos`. Do not run the iOS binary on the host or install
a simulator as part of this build.
