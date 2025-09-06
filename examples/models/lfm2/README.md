## Summary
[LFM2](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38) is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/) and available in 3 variants - 350M, 700M, 1.2B.

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
### Example run
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

With ExecuTorch's sample c++ runner (see the Llama README's [Step 3: Run on your computer to validate](../llama/README.md#step-3-run-on-your-computer-to-validate) to build the runner):
```
cmake-out/examples/models/llama/llama_main \
  --model_path lfm2_700m_8da4w.pte \
  --tokenizer_path ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M/snapshots/ab260293733f05dd4ce22399bea1cae2cf9b272d/tokenizer.json \
  --prompt="<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n" \
  --temperature 0.3
```

To run the model on an example iOS or Android app, see the Llama README's [Step 5: Build Mobile apps](../llama/README.md#step-5-build-mobile-apps) section.
