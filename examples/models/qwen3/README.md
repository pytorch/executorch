## Summary
Qwen 3 is the latest iteration of the Qwen series of large language models (LLMs) developed by Alibaba. Edge-sized Qwen3 model variations (0.6B, 1.7B, and 4B) are currently supported .

## Instructions

Qwen 3 uses the same example code as our optimized Llama model, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.

All commands for exporting and running Llama on various backends should also be applicable to Qwen 3, by swapping the following args:
```
base.model_class=[qwen3_0_6b,qwen3_1_7b,qwen3_4b]
base.params=[examples/models/qwen3/config/0_6b_config.json,examples/models/qwen3/config/1_7b_config.json,examples/models/config/qwen3/4b_config.json]
```

### Example export
Here is a basic example for exporting Qwen 3, although please refer to the Llama README's [Step 2: Prepare model](../llama/README.md#step-2-prepare-model) for more advanced usage.

Export 0.6b to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3/config/qwen3_xnnpack_q8da4w.yaml \
  +base.model_class="qwen3_0_6b" \
  +base.params="examples/models/qwen3/config/0_6b_config.json" \
  +export.output_name="qwen3_0_6b.pte"

```

Export 1.7b to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3/config/qwen3_xnnpack_q8da4w.yaml \
  +base.model_class="qwen3_1_7b" \
  +base.params="examples/models/qwen3/config/1_7b_config.json" \
  +export.output_name="qwen3_1_7b.pte"
```

Export 4b to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3/config/qwen3_xnnpack_q8da4w.yaml \
  +base.model_class="qwen3_4b" \
  +base.params="examples/models/qwen3/config/4b_config.json" \
  +export.output_name="qwen3_4b.pte"
```

### Example run
With ExecuTorch pybindings:
```
python -m examples.models.llama.runner.native \
  --model qwen3_0_6b \
  --pte qwen3_0_6b.pte \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer.json \
  --tokenizer_config ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer_config.json \
  --prompt "Who is the president of the US?" \
  --params examples/models/qwen3/config/0_6b_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.6
```

With ExecuTorch's sample c++ runner (see the Llama README's [Step 3: Run on your computer to validate](../llama/README.md#step-3-run-on-your-computer-to-validate) to build the runner):
```
cmake-out/examples/models/llama/llama_main \
  --model_path qwen3_0_6b.pte \
  --tokenizer_path ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer.json \
  --prompt="<|im_start|>user Who is the president of the US?<|im_end|><|im_start|>assistant"
```
Note that you have to apply the chat template manually for the C++ runner.

To run the model on an example iOS or Android app, see the Llama README's [Step 5: Build Mobile apps](../llama/README.md#step-5-build-mobile-apps) section.

### DFlash speculative decoding (MLX delegate)

`export_dflash_draft.py`, `run_dflash.py`, and `run_baseline.py` implement
block-diffusion speculative decoding (DFlash) for Qwen3 on the MLX delegate.
See `mlx_source_transformations.py` for the hidden-state-tapping wrapper used
during export.

The `check_dflash_*.py` scripts under `tests/` are manual driver scripts, not
pytest tests -- they require exported `qwen3_4b_dflash_target.pte` /
`_draft.pte` files (multi-GB, not checked in), HF downloads, and Apple
M-series hardware with the MLX delegate, so they cannot run in this repo's
CI. Run them by hand after exporting:

```bash
python examples/models/qwen3/tests/check_dflash_target.py qwen3_4b_dflash_target.pte
python examples/models/qwen3/tests/check_dflash_draft.py qwen3_4b_dflash_draft.pte
python examples/models/qwen3/tests/check_dflash_lossless.py
```

The "lossless" guarantee (DFlash output is token-for-token identical to
greedy baseline decoding) is currently only verified this way, manually.

### FAQ
For more help with exporting or running this model, feel free to ask in our [discord channel](https://discord.gg/UEjkY9Zs).
