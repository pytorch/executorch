## Summary
Qwen 2.5 is the latest iteration of the Qwen series of large language models (LLMs) developed by Alibaba. At the moment, 1.5b is currently supporting, with plans in the future for adding the 0.5b and 3b versions.

## Instructions

Qwen 2.5 uses the same example code as Llama, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.

All commands for exporting and running Llama on various backends should also be applicable to Qwen 2.5, by swapping the following args:

```
base.model_class=[qwen2_5_0_5b, qwen2_5_1_5b]
base.params=[examples/models/qwen2_5/config/0_5b_config.json, examples/models/qwen2_5/config/1_5b_config.json]
base.checkpoint=<path-to-meta-checkpoint>
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
Here is an basic example for exporting and running Qwen 2.5, although please refer to [Llama README page](../llama/README.md) for more advanced usage.

Export to XNNPack, no quantization:
```
# No quantization
# Set these paths to point to the downloaded files
QWEN_CHECKPOINT=path/to/checkpoint.pth

python -m extension.llm.export.export_llm \
  --config examples/models/qwen2_5/config/qwen2_5_xnnpack_q8da4w.yaml \
  +base.model_class="qwen2_5_1_5b" \
  +base.checkpoint="${QWEN_CHECKPOINT:?}" \
  +base.params="examples/models/qwen2_5/config/1_5b_config.json" \
  +export.output_name="qwen2_5-1_5b.pte" \
```

Run using the executor runner:
```
# Currently a work in progress, just need to enable HuggingFace json tokenizer in C++.
# In the meantime, can run with an example Python runner with pybindings:

python -m examples.models.llama.runner.native \
  --model qwen2_5 \
  --pte <path-to-pte> \
  -kv \
  --tokenizer <path-to-tokenizer>/tokenizer.json \
  --tokenizer_config <path-to_tokenizer>/tokenizer_config.json \
  --prompt "Who is the founder of Meta?" \
  --params examples/models/qwen2_5/config/1_5b_config.json \
  --max_len 64 \
  --temperature 0
```
