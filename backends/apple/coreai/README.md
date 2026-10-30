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
