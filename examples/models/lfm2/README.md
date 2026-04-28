## Summary
[LFM2](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38) is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/) and available in 3 variants - 350M, 700M, 1.2B.

[LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) is an updated version with improved training (28T tokens vs 10T) and extended context length support (32K tokens).

## Instructions

LFM2 uses the same example code as optimized Llama model, while the checkpoint, model params, and tokenizer are different. Please see the [Llama README page](../llama/README.md) for details.
LFM2 is a hybrid model, where some attention layers are replaced with short convolutions.

### Example export
Here is a basic example for exporting LFM2, although please refer to the Llama README's [Step 2: Prepare model](../llama/README.md#step-2-prepare-model) for more advanced usage.

Export 350m to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_xnnpack_q8da4w.yaml \
  +base.model_class="lfm2_350m" \
  +base.params="examples/models/lfm2/config/lfm2_350m_config.json" \
  +export.output_name="lfm2_350m_8da4w.pte"
```

Export 700m to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_xnnpack_q8da4w.yaml \
  +base.model_class="lfm2_700m" \
  +base.params="examples/models/lfm2/config/lfm2_700m_config.json" \
  +export.output_name="lfm2_700m_8da4w.pte"
```

Export 1_2b to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_xnnpack_q8da4w.yaml \
  +base.model_class="lfm2_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_1_2b_config.json" \
  +export.output_name="lfm2_1_2b_8da4w.pte"
```

Export LFM2.5 1.2B to XNNPack, quantized with 8da4w:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_xnnpack_q8da4w.yaml \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  +export.output_name="lfm2_5_1_2b_8da4w.pte"
```

Export LFM2.5 350M to MLX on Apple Silicon, quantized with 4-bit weights:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_mlx_4w.yaml \
  +base.model_class="lfm2_5_350m" \
  +base.params="examples/models/lfm2/config/lfm2_5_350m_config.json" \
  +export.output_name="lfm2_5_350m_mlx_4w.pte"
```

Export LFM2.5 1.2B to MLX on Apple Silicon, quantized with 4-bit weights:
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_mlx_4w.yaml \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  +export.output_name="lfm2_5_1_2b_mlx_4w.pte"
```

To export with extended context (e.g., 2048 tokens):
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_xnnpack_q8da4w.yaml \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  +export.max_seq_length=2048 \
  +export.max_context_length=2048 \
  +export.output_name="lfm2_5_1_2b_8da4w.pte"
```
### Example run
For MLX on Apple Silicon, build or install ExecuTorch with MLX enabled. The
easiest local path is:
```
conda activate et-mlx
python install_executorch.py
xcrun -sdk macosx --find metal
```

The `metal` command must resolve to an Xcode path, not fail under standalone
Command Line Tools.

With ExecuTorch pybindings:
```
python -m examples.models.llama.runner.native \
  --model lfm2_700m \
  --pte lfm2_700m_8da4w.pte \
  --tokenizer ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M/snapshots/ab260293733f05dd4ce22399bea1cae2cf9b272d/tokenizer.json \
  --tokenizer_config ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M/snapshots/ab260293733f05dd4ce22399bea1cae2cf9b272d/tokenizer_config.json \
  --prompt "<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n" \
  --params examples/models/lfm2/config/lfm2_700m_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.3
```

With ExecuTorch pybindings and an LFM2.5 MLX export:
```
python -m examples.models.llama.runner.native \
  --model lfm2_5_350m \
  --pte lfm2_5_350m_mlx_4w.pte \
  --tokenizer ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/<snapshot>/tokenizer.json \
  --tokenizer_config ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/<snapshot>/tokenizer_config.json \
  --prompt "<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n" \
  --params examples/models/lfm2/config/lfm2_5_350m_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.3
```

Find the Hugging Face cache snapshot directory with:
```
python - <<'PY'
from pathlib import Path
root = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
for path in root.glob("*/tokenizer.json"):
    print(path.parent)
PY
```

With ExecuTorch's sample c++ runner (see the Llama README's [Step 3: Run on your computer to validate](../llama/README.md#step-3-run-on-your-computer-to-validate) for general runner details):
```
cmake-out/examples/models/llama/llama_main \
  --model_path lfm2_700m_8da4w.pte \
  --tokenizer_path ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M/snapshots/ab260293733f05dd4ce22399bea1cae2cf9b272d/tokenizer.json \
  --prompt="<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n" \
  --temperature 0.3
```

Build the C++ runner with MLX support for LFM2.5:
```
make lfm_2_5-mlx
```

Then run an LFM2.5 MLX export with the C++ runner:
```
cmake-out/examples/models/llama/llama_main \
  --model_path lfm2_5_350m_mlx_4w.pte \
  --tokenizer_path ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/<snapshot>/tokenizer.json \
  --prompt="<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n" \
  --temperature 0.3
```

To run the model on an example iOS or Android app, see the Llama README's [Step 5: Build Mobile apps](../llama/README.md#step-5-build-mobile-apps) section.
