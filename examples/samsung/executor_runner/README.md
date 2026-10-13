# Exynos Executor Runner Configuration

This directory contains the executor runner for LLMs on Exynos platforms. The runner requires a JSON configuration file to correctly set up the model architecture and runtime parameters.

## Building

The LLM runner is not part of the default example build. After building the backend with `./backends/samsung/build.sh`, build it with:

```bash
./examples/samsung/build.sh --llm
```

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

For Gemma 4 multimodal inference, also pass the vision encoder and the image with `--vision_model_path`, `--image_path` and, optionally, `--prompt_before_image` for the text placed before the image.

### Where to find config files

Configuration files for supported models are located in `configs/`:
- Gemma 3 1B: `configs/exynos_gemma3_1b.json`
- Gemma 4 E2B: `configs/exynos_gemma4_e2b.json`
- Llama 3 1B: `configs/exynos_llama3_1b.json`

Ensure these files are pushed to the target device (e.g., to `/data/local/tmp/`) before running the script.

### Position embeddings

`--pos_embedding_path` is optional: when the file is missing, the runner precomputes the RoPE tables from the config. For Gemma 4, generate the table that matches the exported model with:

```bash
python -m executorch.examples.samsung.scripts.gemma4_pos_embedding \
    --config <hf_config.json> --max-seq-len 4096 --out pos_embedding.bin
```

## `exynos_llm_wrapper` library (app-porting reference)

`libexynos_llm_wrapper.so` is a reference for porting the Exynos LLM runner into
an Android app: a small public C API (`exynos_llm_wrapper.h`) that hides the
runner internals behind an opaque handle. `exynos_llm_wrapper_sample` is a CLI
that exercises the same API on-device.

`enn_executor_llm_runner` always drives the runner through this C API, but the
`.so` and the sample are only produced when requested:

```bash
./examples/samsung/build.sh --llm-wrapper-so
```
