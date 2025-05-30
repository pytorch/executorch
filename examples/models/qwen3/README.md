## Summary
Qwen 3 is the latest iteration of the Qwen series of large language models (LLMs) developed by Alibaba. Edge-sized Qwen3 model variations (0.6B, 1.7B, and 4B) are currently supported .

## Instructions

Qwen 3 uses the same example code as our optimized Llama model, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.

All commands for exporting and running Llama on various backends should also be applicable to Qwen 3, by swapping the following args:
```
--model [qwen3-0.6b,qwen3-1_7b,qwen3-4b]
--params [examples/models/qwen3/0_6b_config.json,examples/models/qwen3/1_7b_config.json,examples/models/qwen3/4b_config.json]
```

### Example export
Here is a basic example for exporting Qwen 3, although please refer to the Llama README's [Step 2: Prepare model](../llama/README.md#step-2-prepare-model) for more advanced usage.

Export 0.6b to XNNPack, quantized with 8da4w:
```
python -m examples.models.llama.export_llama \
  --model qwen3-0_6b \
  --params examples/models/qwen3/0_6b_config.json \
  -kv \
  --use_sdpa_with_kv_cache \
  -d fp32 \
  -X \
  --xnnpack-extended-ops \
  -qmode 8da4w \
  --metadata '{"get_bos_id": 151644, "get_eos_ids":[151645]}' \
  --output_name="qwen3-0_6b.pte" \
  --verbose
```

Export 1.7b to XNNPack, quantized with 8da4w:
```
python -m examples.models.llama.export_llama \
  --model qwen3-1_7b \
  --params examples/models/qwen3/1_7b_config.json \
  -kv \
  --use_sdpa_with_kv_cache \
  -d fp32 \
  -X \
  --xnnpack-extended-ops \
  -qmode 8da4w \
  --metadata '{"get_bos_id": 151644, "get_eos_ids":[151645]}' \
  --output_name="qwen3-1_7b.pte" \
  --verbose
```

Export 4b to XNNPack, quantized with 8da4w:
```
python -m examples.models.llama.export_llama \
  --model qwen3-4b \
  --params examples/models/qwen3/4b_config.json \
  -kv \
  --use_sdpa_with_kv_cache \
  -d fp32 \
  -X \
  --xnnpack-extended-ops \
  -qmode 8da4w \
  --metadata '{"get_bos_id": 151644, "get_eos_ids":[151645]}' \
  --output_name="qwen3-4b.pte" \
  --verbose
```

### Example run
With ExecuTorch pybindings:
```
python -m examples.models.llama.runner.native
  --model qwen3-0_6b \
  --pte qwen3-0_6b.pte \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer.json \
  --tokenizer_config ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer_config.json \
  --prompt "Who is the president of the US?" \
  --params examples/models/qwen3/0_6b_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.6
```

With ExecuTorch's sample c++ runner (see the Llama README's [Step 3: Run on your computer to validate](../llama/README.md#step-3-run-on-your-computer-to-validate) to build the runner):
```
cmake-out/examples/models/llama/llama_main
  --model_path qwen3-0_6b.pte
  --tokenizer_path ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143/tokenizer.json
  --prompt="Who is the president of the US?"
```

To run the model on an example iOS or Android app, see the Llama README's [Step 5: Build Mobile apps](../llama/README.md#step-5-build-mobile-apps) section.

### FAQ
For more help with exporting or running this model, feel free to ask in our [discord channel](https://discord.gg/UEjkY9Zs).
