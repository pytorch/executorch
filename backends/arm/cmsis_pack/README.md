# PyTorch::ExecuTorch CMSIS Pack

Build scripts and templates for the `PyTorch::ExecuTorch` CMSIS Pack.

## Overview

This is a **source pack**: it ships ExecuTorch runtime + kernel sources
(not a prebuilt binary), packaged as a CMSIS Pack for bare-metal Cortex-M
consumers. Every portable, quantized and Cortex-M operator is exposed as a
selectable CMSIS component, enabling fine-grained code-size control.

## Structure

```
backends/arm/
├── cmsis_pack/
│   ├── config/
│   │   └── executorch_config.yml       # Build configuration and defines
│   ├── contributions/
│   │   ├── add/                        # Static files copied verbatim into the pack
│   │   │   ├── Documentation/
│   │   │   └── armclang_shims/sys/     # AC6-only sys/types.h shim (compiler.h fix)
│   │   └── runtime/platform/default/
│   │       └── minimal.cpp.patch       # Pack-local patch applied to upstream minimal.cpp
│   ├── templates/
│   │   └── PyTorch.ExecuTorch.pdsc.tpl # Pack description template
│   ├── test/
│   │   ├── validate_pack.py            # Structural validation of a built .pack
│   │   └── smoke/                      # csolution consumer-build smoke project
│   │       ├── run.sh                  # Local driver (build + validate + cbuild)
│   │       ├── smoke.csolution.yml
│   │       ├── smoke.cproject.yml
│   │       ├── vcpkg-configuration.json
│   │       └── main.cpp
│   └── scripts/
│       ├── build_pack.sh               # Main entry point
│       ├── copy_sources.sh             # Collects sources from repo tree
│       ├── generate_components.py      # Generates per-operator PDSC components
│       └── generate_register_all_kernels.py  # Generates #ifdef-guarded registrations
```

The build/codegen scripts and the test harness are co-located under
`backends/arm/cmsis_pack/`, keeping everything pack-specific in one tree.

## Components

- **Machine Learning::ExecuTorch::Runtime** — Core runtime (always required)
- **Machine Learning::ExecuTorch::Kernel Utils** — Kernel registration utilities
- **Machine Learning::ExecuTorch::Kernel Registration** — Per-op kernel registrations
- **Machine Learning::ExecuTorch Operators::Portable \*** — Individual portable operators
- **Machine Learning::ExecuTorch Operators::Quantized \*** — Quantized operators
- **Machine Learning::ExecuTorch Operators::Cortex-M \*** — CMSIS-NN-optimized Cortex-M operators
- **Machine Learning::ExecuTorch::Backend EthosU** — Ethos-U NPU backend for Cortex-M host (bare-metal)

## Building locally

```bash
# 1. Cross-compile ExecuTorch for Cortex-M (generates required headers).
#    Equivalent to running backends/arm/scripts/build_executorch.sh — use
#    that script if you want the canonical Arm-backend build flags.
cmake \
  -DCMAKE_TOOLCHAIN_FILE=examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_FLATC=ON \
  -Bcmake-out-arm .
cmake --build cmake-out-arm --config Release -j$(nproc)

# 2. Build the pack. --output-dir is where the .pack archive lands
#    (created if absent); each invocation rewrites this directory.
backends/arm/cmsis_pack/scripts/build_pack.sh \
  --executorch-root "$(pwd)" \
  --build-dir cmake-out-arm \
  --version "$(cat version.txt | sed 's/a0$//')" \
  --output-dir pack-output
```

The resulting `.pack` file is a zip archive installable via `cpackget add <file>.pack`.

## Local testing

Two scripts under `test/` exercise a freshly built pack the way real
consumers will. Both run outside CI today; the CI workflow only builds and
uploads the `.pack` artifact.

```bash
# End-to-end: rebuild the pack, validate its structure, then run a
# csolution + cbuild consumer build inside the AVH-MLOps Docker image.
backends/arm/cmsis_pack/test/smoke/run.sh
```

The driver:

1. Calls `build_pack.sh` to produce a fresh `.pack` (always rebuilds — no
   stale install gets exercised).
2. Runs `test/validate_pack.py` against the archive: PDSC well-formed,
   runtime + `RegisterAllKernels.cpp` present, no duplicate / leaked
   `.py` entries, every `<file name="..."/>` reference resolves.
3. Spawns `ghcr.io/arm-software/avh-mlops/arm-mlops-docker-licensed-community:latest-arm64`,
   `vcpkg activate`s the toolchain set declared in
   `test/smoke/vcpkg-configuration.json` (cmsis-toolbox + arm-none-eabi-gcc
   + cmake + ninja), `cpackget`-installs the freshly built pack into a
   container-local pack root, then `cbuild`s the smoke project for the
   `ARMCM55` target. All compile flags come from the PDSC and the
   cmsis-toolbox `cdefault.yml` — none are hand-curated in the script.

Override defaults via env vars (defaults shown in parentheses):

| Var | Default | Meaning |
|-----|---------|---------|
| `PACK_VERSION` | `<version.txt>-stage` | Version string baked into the pack |
| `BUILD_DIR` | `arm_test/cmake-out` | CMake build dir feeding generated headers |
| `OUTPUT_DIR` | `arm_test/cmsis-pack-output` | Where the `.pack` archive lands |
| `DOCKER_IMAGE` | `ghcr.io/arm-software/avh-mlops/arm-mlops-docker-licensed-community:latest-arm64` | Image used for the consumer build |

Prerequisite: `BUILD_DIR` must be populated by
`backends/arm/scripts/build_executorch.sh` so the generated FlatBuffers /
schema headers are available to `build_pack.sh`.

To validate a previously built `.pack` archive without rebuilding or
running the consumer build:

```bash
python3 backends/arm/cmsis_pack/test/validate_pack.py path/to/PyTorch.ExecuTorch.<ver>.pack
```

## Dependencies

- ARM::CMSIS
- ARM::CMSIS-NN (for CMSIS-NN optimized operators)
- ARM::ethos-u-core-driver (for Ethos-U backend)
