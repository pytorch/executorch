## Summary
Phi-4-mini Instruct (3.8B) is a newly released version of the popular Phi-4 model developed by Microsoft.

## Instructions

Phi-4-mini uses the same example code as Llama, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.

All commands for exporting and running Llama on various backends should also be applicable to Phi-4-mini, by swapping the following args:
```
--model phi_4_mini
--params examples/models/phi-4-mini/config.json
--checkpoint <path-to-meta-checkpoint>
```

### Generate the Checkpoint
The original checkpoint can be obtained from HuggingFace:
```
huggingface-cli download microsoft/Phi-4-mini-instruct
```

We then convert it to Meta's checkpoint format:
```
python examples/models/phi-4-mini/convert_weights.py <path-to-checkpoint-dir> <output-path>
```

### Example export and run
Here is an basic example for exporting and running Phi-4-mini, although please refer to [Llama README page](../llama/README.md) for more advanced usage.

Export to XNNPack, no quantization:
```
# No quantization
# Set these paths to point to the downloaded files
PHI_CHECKPOINT=path/to/checkpoint.pth

python -m examples.models.llama.export_llama \
  --model phi_4_mini \
  --checkpoint "${PHI_CHECKPOINT=path/to/checkpoint.pth:?}" \
  --params examples/models/phi-4-mini/config.json \
  -kv \
  --use_sdpa_with_kv_cache \
  -d fp32 \
  -X \
  --metadata '{"get_bos_id":151643, "get_eos_ids":[151643]}' \
  --output_name="phi-4-mini.pte"
  --verbose
```

Run using the executor runner:
```
# Currently a work in progress, just need to enable HuggingFace json tokenizer in C++.
# In the meantime, can run with an example Python runner with pybindings:

python -m examples.models.llama.runner.native
  --model phi_4_mini
  --pte <path-to-pte>
  -kv
  --tokenizer <path-to-tokenizer>/tokenizer.json
  --tokenizer_config <path-to_tokenizer>/tokenizer_config.json
  --prompt "What is in a california roll?"
  --params examples/models/phi-4-mini/config.json
  --max_len 64
  --temperature 0
```
