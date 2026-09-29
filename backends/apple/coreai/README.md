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

## Export Manifest

The exporter writes a manifest into each delegate's processed bytes with
`version: 2`, a `function` name, and ordered `input_names` and `output_names`
matching the converter's bindings. Dictionary iteration order is not a binding
contract. The `files` object maps relative filenames to byte sizes;
`bundle_digests` maps each bundle basename to its export-time SHA-256 digest.

AOT manifests include a target platform and an exact architecture map. These
are Core AI architecture names, not CPU names such as `arm64`.

Enumerated input shapes produce one statically shaped function per entry, so
those manifests carry a `functions` list (each `main_<key>` name with its input
shapes) instead of `function`, and are marked `runtime_supported: false` until
the runtime can select a function by input shape. Programs with delegate-owned
mutated buffers are also marked `runtime_supported: false`; mutated user inputs
are rejected at export.
