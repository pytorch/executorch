---
name: zephyr
description: Build and configure ExecuTorch as a Zephyr RTOS module for embedded boards. Use when setting up a Zephyr workspace with ET, adding board support (overlays, confs, memory layout), building with west, or debugging linker memory overflow.
---

# ExecuTorch on Zephyr

## When to use this skill

- Setting up a Zephyr workspace with ExecuTorch as a module
- Adding or configuring a board for an ET Zephyr sample
- Debugging `west build` failures (linker overflow, section placement, CMake/Python issues)
- Sizing allocator pools for a specific model + board combination

## When to use a different skill

| Need | Skill |
|------|-------|
| Export a model to .pte | `/export` |
| Bare-metal Cortex-M (no RTOS) | `/cortex-m` |
| General ET C++ build (not Zephyr) | `/building` |
| Backend op support / known issues | `/executorch-kb` |

## Advanced Topics

| Topic | File | When to read |
|-------|------|--------------|
| Adding a new board | `board_bringup.md` | User wants to add overlay, conf, linker snippets for a new board |
| Memory overflow debugging | `memory_debugging.md` | Build fails with region overflow, or runtime allocation failure |

## Architecture

ExecuTorch integrates as a **Zephyr external module** via `zephyr/module.yml`. The module exposes ET libraries (runtime, kernels, backends) as Zephyr CMake targets that applications link against.

```
zephyr_workspace/
├── zephyr/                      # Zephyr kernel
│   └── submanifests/
│       └── executorch.yaml      # pulls ET as a west project
├── modules/lib/executorch/      # ET source (or symlink for dev)
│   └── zephyr/
│       ├── module.yml           # declares ET as Zephyr module
│       ├── CMakeLists.txt       # top-level Zephyr-aware build
│       └── samples/
│           ├── hello-executorch/
│           └── mv2-ethosu/
└── build/                       # west build output
```

## Setup

### 1. Create Zephyr workspace

```bash
mkdir zephyr_workspace && cd zephyr_workspace
python3 -m venv .venv && source .venv/bin/activate
pip install west "cmake<4.0.0" pyelftools ninja jsonschema

west init --manifest-rev v4.3.0
```

### 2. Add ExecuTorch as a module

Create `zephyr/submanifests/executorch.yaml` with the manifest snippet (see
`zephyr/README.md` in the ET repo for the canonical content), or copy it from
an existing ET checkout:

```bash
# From an existing ET checkout (before west update):
cp /path/to/your/executorch/zephyr/executorch.yaml zephyr/submanifests/
```

For local development, symlink your ET checkout after `west update`:

```bash
west config manifest.project-filter -- '-.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u'
west update
rm -rf modules/lib/executorch
ln -s /path/to/your/executorch modules/lib/executorch
```

### 3. Install ExecuTorch

```bash
cd modules/lib/executorch
git submodule sync && git submodule update --init --recursive
./install_executorch.sh
cd ../../..
```

### 4. Install Zephyr SDK

```bash
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/zephyr-sdk-0.17.4_linux-x86_64_minimal.tar.xz
tar xf zephyr-sdk-0.17.4_linux-x86_64_minimal.tar.xz
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz
tar xf toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz -C zephyr-sdk-0.17.4/
./zephyr-sdk-0.17.4/setup.sh -c -t arm-zephyr-eabi
export ZEPHYR_SDK_INSTALL_DIR=$(realpath ./zephyr-sdk-0.17.4)
```

### 5. Install Ethos-U tools (if targeting NPU boards)

```bash
modules/lib/executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula
source modules/lib/executorch/examples/arm/arm-scratch/setup_path.sh
```

This installs Vela compiler and Corstone FVP binaries.

## Building

### Basic build

```bash
west build -b <board> modules/lib/executorch/zephyr/samples/<sample> -- \
    -DET_PTE_FILE_PATH=<path/to/model.pte>
```

### Build and run on FVP

```bash
west build -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/mv2-ethosu -t run -- \
    -DET_PTE_FILE_PATH=mv2_u55_128.pte
```

### Force correct Python

If CMake picks up the wrong Python (common on systems with multiple interpreters):

```bash
west build ... -- -DPython3_EXECUTABLE=$(which python3)
```

### Clean rebuild

```bash
rm -rf build && west build ...
```

## ET-Specific Zephyr Concepts

### Model embedding

`pte_to_header.py` converts a `.pte` file into a C header with the model bytes placed in a named section (default: `network_model_sec`). The section name is controlled by `ET_PTE_SECTION` in CMakeLists.txt.

### Allocator pools

ET requires three memory pools at runtime. The method and temp pools are sized via Zephyr Kconfig; fast scratch is a compile-time macro in some samples.

| Pool | Setting | Purpose |
|------|---------|---------|
| Method allocator | `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` (Kconfig) | Planned buffers + input/output tensors |
| Temp allocator | `CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE` (Kconfig) | Delegate scratch (e.g., Ethos-U scratch buffer) |
| Fast scratch | `ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE` (compile macro) | Small Ethos-U fast memory |

Defaults are sample- and model-dependent — check the specific sample's `prj.conf`, board `.conf`, and Kconfig definitions rather than assuming a single repo-wide default.

Pool sizing depends on the model. To find required sizes:
1. Build and run — runtime errors report exactly how many bytes were requested vs available
2. The method pool must hold the largest planned buffer + all input tensors
3. The temp pool must hold the delegate's scratch buffer (model-dependent)
4. If used, fast scratch must cover the backend's small fast-memory requirement

### DMA accessibility

NPU backends (Ethos-U) require model data and scratch buffers in DMA-accessible memory. Which regions are DMA-accessible depends on the board:

| Board | DMA-accessible regions |
|-------|----------------------|
| Corstone-300 FVP | ISRAM (0x31000000), DDR (0x60000000+) |
| Corstone-320 FVP | ISRAM (0x31000000), DDR (0x70000000+) |
| Alif Ensemble | MRAM (model in-place), SRAM |

When `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y` is set in the board conf (which defines `ET_ARM_MODEL_PTE_DMA_ACCESSIBLE` for the C++ sources), the runtime skips copying the model blob to a writable SRAM buffer. Use this when the model already resides in DMA-accessible memory (DDR, MRAM). Note: this Kconfig symbol is defined per-sample (e.g., in `mv2-ethosu/Kconfig`), not globally.

### Selective ops build

`gen_oplist.py` reads the .pte file to determine which ops are needed and generates a selective kernel build. If the model is fully NPU-delegated, no portable ops are built. If fallback ops exist (e.g., `aten::` ops not handled by the delegate), only those specific kernels are compiled.

## Key Files

| File | Purpose |
|------|---------|
| `zephyr/module.yml` | Declares ET as a Zephyr module |
| `zephyr/CMakeLists.txt` | Top-level Zephyr-aware CMake (builds ET libs as Zephyr targets) |
| `zephyr/Kconfig` | Root Kconfig for ET module (build options, portable ops toggle) |
| `zephyr/executorch.yaml` | West submanifest — pulls ET + dependencies |
| `zephyr/samples/*/CMakeLists.txt` | Per-sample build (model embedding, op selection, pool sizing) |
| `zephyr/samples/*/boards/*.overlay` | Board-specific DTS overlays (memory regions, chosen nodes) |
| `zephyr/samples/*/boards/*.conf` | Board-specific Kconfig (drivers, pool sizes, DMA flags) |
| `codegen/tools/gen_oplist.py` | Reads .pte to generate selective op list |
| `examples/arm/executor_runner/pte_to_header.py` | Converts .pte to C header with section attribute |

## Supported Boards

| Board | NPU | Memory | Status |
|-------|-----|--------|--------|
| mps3/corstone300/fvp | Ethos-U55 | 512K ITCM + 2M ISRAM | hello-executorch (full), MV2 (build-only) |
| mps4/corstone320/fvp | Ethos-U85 | 4M ISRAM + 2M SRAM | hello-executorch (full), MV2 (full) |
| alif_e8_dk | Ethos-U55-256 | 4M MRAM + 2M SRAM | MV2 (model runs from MRAM in-place) |
