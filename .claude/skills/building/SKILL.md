---
name: building
description: Build ExecuTorch runners or C++ libraries. Use when compiling runners for Llama, Whisper, or other models, or building the C++ runtime.
---

# Building

## Runners (Makefile)
```bash
make help              # list all targets
make llama-cpu         # Llama
make whisper-metal     # Whisper on Metal
make gemma3-cuda       # Gemma3 on CUDA
```

Output: `cmake-out/examples/models/<model>/<runner>`

## C++ Libraries (CMake)
```bash
cmake --list-presets                    # list presets
cmake --workflow --preset llm-release   # LLM CPU
cmake --workflow --preset llm-release-metal  # LLM Metal
```
