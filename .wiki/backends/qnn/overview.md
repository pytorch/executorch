---
title: QNN Backend Overview
category: BACKEND_CONSTRAINT
backends: [QNN]
socs: [SM8450, SM8475, SM8550, SM8650, SM8750, SA8255, SA8295, SA8797, SXR2230P, QCM6490]
last_validated: 2026-04-05
source_issues: [1176, 3586, 3949, 4973, 5199, 5914, 8640, 10281, 10895, 10966, 10993, 11100, 11807, 15387, 16535, 16465]
---

# QNN Backend Overview

## What is the QNN Backend?

The QNN (Qualcomm AI Engine Direct) backend delegates model execution to Qualcomm's Hexagon NPU (HTP - Hexagon Tensor Processor) on Snapdragon SoCs. It supports ahead-of-time (AOT) compilation on an x86 host, producing context binaries embedded in `.pte` files that execute on-device via the HTP runtime. [Source: #3586, #8640]

## Hardware Targets

The QNN backend targets Qualcomm Snapdragon SoCs across mobile, automotive, XR, and IoT platforms:

| Category | SoCs | HTP Arch | Notes |
|----------|------|----------|-------|
| Mobile | SM8450 (8 Gen 1), SM8475 (8+ Gen 1) | V69 | No 16-bit matmul 2nd input (use `annotate_kv_8bit` for LLMs) [Source: #15410, #16690, #17296] |
| Mobile | SM8550 (8 Gen 2) | V73 | First arch with 16-bit matmul [Source: #4973] |
| Mobile | SM8650 (8 Gen 3) | V75 | Full feature support [Source: #4973] |
| Mobile | SM8750 (8 Elite) | V79 | Latest mobile, weight sharing support [Source: #16465] |
| Automotive | SA8255 | V73 | Automotive variant [Source: #16217] |
| Automotive | SA8295 | V68 | Oldest supported arch, significant limitations [Source: #1176] |
| Automotive | SA8797 | V81 | 16 MB VTCM; requires QNN SDK v2.42+ [Source: #16535] |
| XR | SXR2230P (Quest 3) | V69 | Same constraints as V69 mobile [Source: #16690] |
| IoT | QCM6490 | V68 | [Source: #16616] |

## High-Level Architecture

### Compilation Flow (AOT on x86 Host)

```
torch.export.export(model)
    → prepare_pt2e(model, QnnQuantizer)  # quantization
    → convert_pt2e(model)
    → to_edge_transform_and_lower_to_qnn(model, inputs, compile_spec)
        → QNN Partitioner (op validation per SoC)
        → QNN Backend preprocess (context binary generation)
    → .pte file with embedded context binaries
```

[Source: #8640, #5199]

### Runtime Flow (On-Device)

1. Load `.pte` file containing QNN context binaries
2. QNN backend restores context from binary (no recompilation)
3. Execute on HTP via FastRPC to Hexagon DSP
4. Results returned to CPU

### Key Components

- **QnnQuantizer**: Handles quantization annotation for QNN-compatible schemes [Source: #1182]
- **QnnPartitioner**: Validates ops against target SoC capabilities, partitions graph [Source: #4973]
- **QnnCompileSpec**: Configures backend options (SoC, optimization level, debug flags) [Source: #3949]
- **Context Binary**: Pre-compiled HTP graph serialized into `.pte` [Source: #8640]

## Two LLM Export Paths

There are two code paths for exporting LLMs to QNN. **Use `examples/qualcomm/oss_scripts/llama/`** — the other path (`examples/models/llama/`) is outdated and produces incorrect results including oversized `.pte` files. [Source: #10226, #11100]

```bash
# CORRECT path for QNN LLM export
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -m SM8650 \
  --decoder_model qwen3-0_6b \
  --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 \
  --prompt "..." --tasks wikitext --limit 1 --compile_only
```

The QNN LLM path uses a custom runner (`qnn_llama_runner`) — the standard `llama_main` runner is incompatible with QNN-exported models. [Source: #11100]

## Supported Backends Within QNN

| Backend | Status | Notes |
|---------|--------|-------|
| HTP (Hexagon) | Production | Primary backend, fully supported |
| GPU (Adreno) | Experimental | Basic support via PR #12165 [Source: #5914] |
| DSP | Planned | Not yet available [Source: #5914] |
| CPU | N/A | Fallback for non-delegated ops |

## Environment Setup

Required environment variables:
```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang  # host
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

On-device:
```bash
export LD_LIBRARY_PATH=/data/local/tmp/qnn_libs  # CPU-side libs
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn_libs  # Skel libs for HTP
```

Both `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` must be set correctly on the device. Missing skel libraries cause `DspTransport.openSession qnn_open failed` errors. [Source: #1527, #1176]

## Build Instructions

```bash
# Install ExecuTorch
./install_executorch.sh

# Build QNN backend (builds all targets including qnn_llama_runner)
backends/qualcomm/scripts/build.sh
```

Use `backends/qualcomm/scripts/build.sh` rather than manual CMake commands — it handles all dependencies correctly. [Source: #4085, #1602, #16217]

## Verifying the Setup

Run a simple model first to verify the environment:
```bash
python examples/qualcomm/scripts/export_example.py -m add -g --soc SM8650
# Push to device and run with qnn_executor_runner
```
[Source: #15387, #16217]

## Custom Model Support

The QNN backend supports arbitrary PyTorch models, not just the examples in `examples/qualcomm/`. If your model's ops are supported by QNN, you can export and run it. [Source: #10966]

## `--compile_only` Flag

Use `--compile_only` to export, quantize, and compile a `.pte` without running on-device. Useful for testing the compilation pipeline on a host machine without a connected Qualcomm device. [Source: #10993]

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -m SM8650 --compile_only \
  --decoder_model qwen3-0_6b ...
```

## Viewing Runner Output on Android

Runner output goes to logcat, not stdout:
```bash
adb logcat | grep ExecuTorch
```
[Source: #11100]

## Benign Warnings

### "Arch 68 set by custom config is different from arch associated with SoC"

This warning appears during x86 host compilation and is **harmless**. QNN sets a default V68 device config, then overrides it with the user-specified target SoC. The override is correct and does not affect performance. [Source: #10281]

### "QnnContextCustomProtocol expected magic number: 0x5678abcd but get: 0x2000000"

This appears when loading context binaries and indicates the `.pte` was compiled for a different SoC or QNN SDK version. If the model runs successfully, it can be ignored. [Source: #11100]

### "Function not called, PrepareLib isn't loaded!"

Benign QNN warning during model loading. Can be ignored. [Source: #10993]

## C++ Tokenizer Setup for Qwen

When building `qnn_llama_runner` for Qwen models, add `-DSUPPORT_REGEX_LOOKAHEAD=ON` to the CMake build. Without this, the regex-based tokenizer patterns won't work correctly. [Source: #11807]

```bash
# Add to build.sh CMake flags:
-DSUPPORT_REGEX_LOOKAHEAD=ON
```

### Qwen Vocab Size Mismatch

Qwen model config reports `vocab_size=151936` but the tokenizer has 151665 entries. The difference is padding tokens for distributed pretraining — this does not affect inference accuracy. [Source: #11807]
