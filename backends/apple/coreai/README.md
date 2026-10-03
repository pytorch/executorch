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

### AOT architecture selection

Configure AOT export with `AOTCompileConfig`, including the target `platform` and
optional `architectures` list. For example, `AOTCompileConfig(platform="iOS",
architectures=["h17p"])` requests that compiler architecture; omitting the list
lets the compiler emit its supported architectures for the target platform.
These are Core AI architecture names, not CPU names such as `arm64`.

Both formats can require device specialization. Every load first attempts SDK
bookmark restoration. A missing bookmark or SDK-confirmed cache miss falls back
to source materialization and persistent specialization, then binds the requested
function and ordered I/O on that same acquired model.

### Asset storage

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
