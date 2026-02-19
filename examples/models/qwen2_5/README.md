## Summary
Qwen 2.5 is the latest iteration of the Qwen series of large language models (LLMs) developed by Alibaba. Supported model variations: 0.5B, 1.5B, and Coder-32B-Instruct.

## Instructions

Qwen 2.5 uses the same example code as Llama, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.

All commands for exporting and running Llama on various backends should also be applicable to Qwen 2.5, by swapping the following args:

```
base.model_class=[qwen2_5_0_5b, qwen2_5_1_5b, qwen2_5_coder_32b]
base.params=[examples/models/qwen2_5/config/0_5b_config.json, examples/models/qwen2_5/config/1_5b_config.json, examples/models/qwen2_5/config/coder_32b_config.json]
```

### Generate the Checkpoint
The original checkpoint can be obtained from HuggingFace:
```
huggingface-cli download Qwen/Qwen2.5-1.5B
```

We then convert it to Meta's checkpoint format:
```
python examples/models/qwen2_5/convert_weights.py <path-to-checkpoint-dir> <output-path>
```

### Example export and run
Here is a basic example for exporting and running Qwen 2.5, although please refer to [Llama README page](../llama/README.md) for more advanced usage.

Export 1.5B to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen2_5/config/qwen2_5_xnnpack_q8da4w.yaml \
  +base.model_class="qwen2_5_1_5b" \
  +base.params="examples/models/qwen2_5/config/1_5b_config.json" \
  +export.output_name="qwen2_5_1_5b.pte"
```

Export Coder-32B to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen2_5/config/qwen2_5_coder_xnnpack_q8da4w.yaml \
  +base.model_class="qwen2_5_coder_32b" \
  +base.params="examples/models/qwen2_5/config/coder_32b_config.json" \
  +export.output_name="qwen2_5_coder_32b.pte"
```

### Example run
With ExecuTorch pybindings:
```
python -m examples.models.llama.runner.native \
  --model qwen2_5_1_5b \
  --pte qwen2_5_1_5b.pte \
  -kv \
  --tokenizer <path-to-tokenizer>/tokenizer.json \
  --tokenizer_config <path-to-tokenizer>/tokenizer_config.json \
  --prompt "Who is the founder of Meta?" \
  --params examples/models/qwen2_5/config/1_5b_config.json \
  --max_len 64 \
  --temperature 0
```
