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
export-time SHA-256 digest. Asset paths, including bundle and directory names,
must remain distinct under case-insensitive, canonically normalized Unicode
comparison. Accepted names are preserved, not rewritten.

Parsing returns an immutable `Manifest` covering all exported bundles. Pure
platform and architecture selection produces an immutable `SelectedBundle`
with one bundle's path, digest, and files. Neither type can be constructed
without validation; filesystem materialization and SDK host probing remain
outside the manifest parser.

### AOT architecture selection

Configure AOT export with `AOTCompileConfig`, including the target `platform` and
optional `architectures` list. For example, `AOTCompileConfig(platform="iOS",
architectures=["h17p"])` requests that compiler architecture; omitting the list
lets the compiler emit its supported architectures for the target platform.
These are Core AI architecture names, not CPU names such as `arm64`.

### Asset storage

The assets root must be an absolute path. The application chooses it, should
reserve it for this backend, and is trusted not to rename, replace or modify its
contents while the backend uses it; the backend does not defend against other
processes of the same user rebinding paths. Written files and changed directories
are synced with `fsync`. Atomic publication uses `F_FULLFSYNC` on the temporary
file before rename and on the directory afterward. A failure before rename
preserves the old contents; a failure of the final flush reports an error even
though the new complete contents are already visible. Use a local APFS volume;
unsupported full-device flushes are reported as errors, not downgraded to `fsync`.

Preparing the assets root sets `NSURLIsExcludedFromBackupKey` on it, which covers
everything beneath it. Ancestors are not modified. This is backup exclusion, not
a control for iCloud Drive synchronization.

Newly created roots request
`NSFileProtectionCompleteUntilFirstUserAuthentication`. On iOS, this allows cache
access while the device is locked after its first unlock following a reboot. New
subdirectories and files inherit their parent's protection. Existing cache
contents are not migrated. Directory preparation skips creation when the root
already exists; child directories are opened first and created only when missing.
Directory creation uses default permissions filtered by the process umask, without
an explicit owner-only restriction. Existing permissions are not changed.
Backup exclusion and required durability flushes still apply on reused paths.

Cache-root preparation owns location and backup/protection policy; descriptor
storage owns no-follow operations, publication, and synchronization. Creation
and enumeration use explicit policies, with a missing child distinguishable
from an error. Fault injection is a separate, thread-local test interface and
compiles away in production.

Each partition derives a key from its selected export digest/bundle, platform,
Core AI device architecture, default SDK cache namespace, versioned default
options and persistent policy. Function bindings and delivery location are
excluded. No model bytes are hashed at runtime. Each key determines a raw
bookmark in the `bookmarks` subdirectory, a bundle directory in `staging`, and an
empty coordination file in `locks`. There is no PTE-wide bookmark list or
serialized lifecycle record.

Raw bookmarks have an 8 MiB backend allocation limit, not an asserted SDK format
maximum, and are published atomically because losing one orphans an SDK entry.
Per-key disk locks coordinate loading and eviction across threads and processes;
different keys remain independent. Lock ownership is maintained by the OS, and
acquiring a lock does not flush its file or directory. Lock files are not
removed by the backend, including after process exit. Removing the assets root
externally requires all loads, sessions and maintenance to stop.

## Host Tests

On macOS, `EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON`
registers the standalone `coreai_host_test` executable with CTest. It tests
manifest parsing, AOT asset selection, and filesystem storage using Foundation,
without Swift or CoreAI SDK calls. The host test target is omitted for Apple
cross-builds.

CI builds and runs these tests on macOS 27. To use the same configuration from
the repository root:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 \
  -DEXECUTORCH_BUILD_COREAI=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
cmake --build <build-dir> --target backends/apple/coreai/all
ctest --test-dir <build-dir>/backends/apple/coreai --output-on-failure --no-tests=error
```
