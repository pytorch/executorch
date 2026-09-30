## Summary
[Spark-X2.5](https://huggingface.co/collections/XHToken/spark-x25) is a compact, efficient general-purpose language model series developed by [XHToken](https://github.com/XHToken/Spark-X2.5), available in two variants — 1.7B and 4B. The models use a hybrid attention architecture that combines one full-attention layer with three sliding-window attention layers, natively supporting context windows of up to 1M tokens and 200+ languages.

## Architecture highlights
- **Hybrid attention**: 3 sliding-window attention layers + 1 full-attention layer (repeating)
- **Per-layer-type RoPE**: full-attention layers use `rope_theta=5M, partial_rotary_factor=0.25`; sliding-attention layers use `rope_theta=10K, partial_rotary_factor=1.0`
- **Headwise attention output gate**: per-head sigmoid gate broadcast over head dim
- **Sliding window**: 512 tokens for sliding-attention layers
- **GELU activation** in the MLP
- **Tied word embeddings**

## Instructions

Spark-X2.5 uses the same export pipeline as the optimized Llama model. Please see the [Llama README](../llama/README.md) for general runner and mobile-app details.

### Example export

Export Spark-X2.5-1.7B to XNNPack, FP32:
```
python -m extension.llm.export.export_llm \
  --config examples/models/spark_x2_5/config/spark_x2_5_xnnpack_fp32.yaml \
  +base.model_class="spark_x2_5_1_7b" \
  +base.params="examples/models/spark_x2_5/config/spark_x2_5_1_7b_config.json" \
  +export.output_name="spark_x2_5_1_7b_fp32.pte"
```

Export Spark-X2.5-1.7B to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/spark_x2_5/config/spark_x2_5_xnnpack_q8da4w.yaml \
  +base.model_class="spark_x2_5_1_7b" \
  +base.params="examples/models/spark_x2_5/config/spark_x2_5_1_7b_config.json" \
  +export.output_name="spark_x2_5_1_7b_8da4w.pte"
```

Export Spark-X2.5-4B to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/spark_x2_5/config/spark_x2_5_xnnpack_q8da4w.yaml \
  +base.model_class="spark_x2_5_4b" \
  +base.params="examples/models/spark_x2_5/config/spark_x2_5_4b_config.json" \
  +export.output_name="spark_x2_5_4b_8da4w.pte"
```

To export with extended context (up to 1024 tokens, bounded by the sliding-window ring cache):
```
python -m extension.llm.export.export_llm \
  --config examples/models/spark_x2_5/config/spark_x2_5_xnnpack_q8da4w.yaml \
  +base.model_class="spark_x2_5_1_7b" \
  +base.params="examples/models/spark_x2_5/config/spark_x2_5_1_7b_config.json" \
  +export.max_seq_length=1024 \
  +export.max_context_length=1024 \
  +export.output_name="spark_x2_5_1_7b_8da4w.pte"
```

Note: the sliding-window attention layers use a ring buffer sized `2 * sliding_window` = 1024 slots, so prefill is bounded to 1024 tokens regardless of `max_seq_length`. The full-attention layers can attend to the full `max_context_length`.

### Example run

Spark-X2.5 uses the following chat template:
```
<｜start▁of▁sentence｜><|System|>
you are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Who are you?<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>
```

> The special tokens `<｜start▁of▁sentence｜>` and `<｜end▁of▁sentence｜>` use the Unicode lower-one-eighth block character (`▁`, U+2581).

With ExecuTorch pybindings:
```
python -m examples.models.llama.runner.native \
  --model spark_x2_5_1_7b \
  --pte spark_x2_5_1_7b_8da4w.pte \
  --tokenizer ~/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/snapshots/<snapshot>/tokenizer.json \
  --tokenizer_config ~/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/snapshots/<snapshot>/tokenizer_config.json \
  --prompt="<｜start▁of▁sentence｜><|System|>
you are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Who are you?<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>" \
  --params examples/models/spark_x2_5/config/spark_x2_5_1_7b_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.3
```

With ExecuTorch's sample C++ runner:
```
cmake-out/examples/models/llama/llama_main \
  --model_path spark_x2_5_1_7b_8da4w.pte \
  --tokenizer_path ~/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/snapshots/<snapshot>/tokenizer.json \
  --prompt="<｜start▁of▁sentence｜><|System|>
you are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Who are you?<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>" \
  --temperature 0.3
```

Find the Hugging Face cache snapshot directory with:
```
python - <<'PY'
from pathlib import Path
root = Path.home() / ".cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/snapshots"
for path in root.glob("*/tokenizer.json"):
    print(path.parent)
PY
```

To run the model on an example iOS or Android app, see the Llama README's [Step 5: Build Mobile apps](../llama/README.md#step-5-build-mobile-apps) section.
