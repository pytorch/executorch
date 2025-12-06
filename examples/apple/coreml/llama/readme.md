# ANE-friendly Llama models

This directory contains ANE-friendly Llama models.

You can export Llama1B to run on the ANE with:
```
python export.py -n /path/to/output/model_dir -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --dtype fp16 --coreml-quantize c4w -E "8,0" --target_split_size 1024 --max_splits 32 --export_in_parts
```

This exports Llama1B in 3 pieces:
* model_dir/input_block.pte (embedding/freq calculations)
* model_dir/transformer_block.pte (transformer layers)
* model_dir/output_block.pte (lm_head)

The model is exported in fp16 and quantized with 4-bit channelwise linear layers, and 8-bit embeddings.  These quantization settings are ANE-friendly, but do require some QAT to get good accuracy.  To run the model, use:

```
buck run :llama_mainAppleMac -- \
  --model_path /path/to/output/model_dir \
  --tokenizer_path /path/to/tokenizer.model \
  --prompt "Once upon a time," \
  --temperature 0.6 \
  --max_new_tokens 100
```

The model is static and predicts 64 tokens at a time, with padding or chunking used to handle sequences of different lengths (this is controlled by seq_length in the export args).  On iOS 26 / iPhone 15 Pro, the prediction time for 64 tokens is 0.03255s, so the approximate performance is:
* Decode: 1 tokens / 0.03255s = 30 tok/sec.
* Prefill: 64 tokens / 0.03255s = 1966 tok/sec

Note that the ANE performance on M1 Mac Pro is much slower than the ANE performance on iPhone 15 Pro, so if you measure on desktop, you should expect worse performance.

### Exporting in one piece

We can also export the model in one piece by leaving off the export arg (--export_in_parts):
```
python export.py -n /path/to/output/model.pte -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --dtype fp16 --coreml-quantize c4w -E "8,0" --target_split_size 1024 --max_splits 8
```

We have observed this leads to significantly slower performance on iPhone.  We do not have a C++ runner for a model that is exported in one piece, but you can test it in python with:

```
python run.py -m /path/to/model.pte -t /path/to/tokenizer.model --prompt "Once upon a time,"
```
