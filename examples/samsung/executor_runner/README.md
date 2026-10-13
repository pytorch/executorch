# Exynos Executor Runner Configuration

This directory contains the executor runner for LLMs on Exynos platforms. The runner requires a JSON configuration file to correctly set up the model architecture and runtime parameters.

## Using JSON Configuration Files

The `enn_executor_llm_runner` binary accepts a path to a JSON configuration file via the `--config_path` argument. This file defines critical model dimensions and strategies used during execution.

### Example Usage

You can refer to `run_llama3_sample.sh` for a complete execution example. The key part of the command is:

```bash
./enn_executor_llm_runner \
    --model_path="/data/local/tmp/llama3/llama3.pte" \
    --config_path="/data/local/tmp/exynos_llama3_1b.json" \
    --prompt="What is the capital of France?" \
    ...
```

### Where to find config files
Configuration files for supported models are located in `examples/models/`. 
For example:
- Gemma 3 1B: `examples/models/gemma3_1b/exynos_gemma3_1b.json`
- Llama 3 1B: `examples/models/llama3_1b/exynos_llama3_1b.json`

Ensure these files are pushed to the target device (e.g., to `/data/local/tmp/`) before running the script.

## `exynos_llm_wrapper` library (app-porting reference)

`libexynos_llm_wrapper.so` is a reference for porting the Exynos LLM runner into
an Android app: a small public C API (`exynos_llm_wrapper.h`) that hides the
runner internals behind an opaque handle. `exynos_llm_wrapper_sample` is a CLI
that exercises the same API on-device.