## Running Hugging Face decoder LLMs on QNN

This directory holds the plumbing that exports a Hugging Face `AutoModelForCausalLM`
to a QNN-delegated `.pte` and runs it with the shared `qnn_llama_runner`.

> ⚠️ **Important:** This is the Hugging Face `transformers` path, *not*
> [the static LLaMA path](../llama/llama.py). The two are independent; model
> definitions, quant recipes and calibration all differ.

There are two entry points, both producing a `.pte` the same runner consumes:

| Script | Flow | Notes |
| --- | --- | --- |
| [`hf_causal_lm.py`](../hf_causal_lm.py) | ExecuTorch-driven (Will be deprecated soon) | Uses `QnnLLMEdgeManager` in [`qnn_decoder_model_manager.py`](./qnn_decoder_model_manager.py). Self-contained — only needs a released `transformers`. |
| [`hf_exporter.py`](../hf_exporter.py) | `transformers`-driven | Calls `transformers.exporters.ExecutorchExporter`, which calls back into [`backends/qualcomm/hf_transformers/api.py`](../../../../backends/qualcomm/hf_transformers/api.py). |

Both share the model wrapper and quant recipes under
`backends/qualcomm/hf_transformers/causal_lm/`, so a recipe change applies to both.

### Prerequisites

1. **Setup ExecuTorch** — follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup).
2. **Setup QNN ExecuTorch** — follow the [tutorial](https://pytorch.org/executorch/main/backends-qualcomm)
   to build the Qualcomm AI Engine Direct backend, and build the runner target
   `examples/qualcomm/oss_scripts/llama/qnn_llama_runner`.
3. **Device access** — ADB reachable device, or `--enable_x86_64` for the x86 emulator.

### Supported models

Model IDs with a tuned quant recipe are listed in `HUGGING_FACE_QUANT_RECIPES`
(`backends/qualcomm/hf_transformers/api.py`). Any other Hugging Face ID still
exports, but falls back to a conservative 16a8w default recipe.

### Example usage

Export and run on device:

```bash
python examples/qualcomm/oss_scripts/hf_causal_lm.py \
    --artifact ./hf_causal_lm \
    --build_folder build-android \
    --decoder_model_id NousResearch/Llama-3.2-1B \
    --prompt "Simply put, the theory of relativity states that" \
    --max_seq_len 128 \
    --soc_model SM8650 \
    --device YOUR_DEVICE_ID \
    --host localhost
```

Useful variations:

*   `--compile_only` — export the `.pte` and stop (no device needed).
*   `--pre_gen_pte` — skip export and run an existing `.pte` from `--artifact`.
*   `-F` / `--use_fp16` — fp16 instead of PTQ.
*   `--enable_x86_64` — run on the x86 emulator instead of a device.

The `transformers`-driven flow takes the same arguments:

```bash
python examples/qualcomm/oss_scripts/hf_exporter.py \
    --artifact ./hf_exporter \
    --build_folder build-android \
    --decoder_model_id NousResearch/Llama-3.2-1B \
    --prompt "Simply put, the theory of relativity states that" \
    --max_seq_len 128 \
    --soc_model SM8650 \
    --device YOUR_DEVICE_ID \
    --host localhost
```

### Output

Generated text is written to `<artifact>/outputs/result.txt` and echoed to the
console. Quantized exports also drop the logits quantization scale/zero_point
next to the `.pte` so a consumer can dequantize the 16-bit logits output.
