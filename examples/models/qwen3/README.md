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

### WebGPU export (Qwen3-0.6B)

`config/qwen3_webgpu_q4gsw.yaml` exports Qwen3-0.6B with group-symmetric 4-bit weights at `max_seq_length` 512 and `max_context_length` 8960, for the WebGPU delegate.

**Runtime selection is declared, not inferred.** `LlmConfig` exposes no WebGPU backend field, so the config sets `backend.vulkan.enabled` — that is the *serialization* mechanism that produces the program the WebGPU delegate consumes (the delegate registers under the Vulkan backend id). Vulkan serialization on its own is not a WebGPU selection. The export contract `manifests/qwen3_0_6b_webgpu.json` therefore records `target_runtime: webgpu` alongside `serialization_backend: vulkan`, and `webgpu_artifact_manifest.py` rejects any artifact set whose declared runtime target is missing or different.

Export:
```
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3/config/qwen3_webgpu_q4gsw.yaml
```

The contract pins the checkpoint and tokenizer to an exact Hugging Face commit, with the published SHA-256 and byte count for each. Build a manifest over an output directory and validate it:
```
python -m executorch.examples.models.qwen3.webgpu_artifact_manifest create \
  --root <output-dir> \
  --output <output-dir>/manifest.json \
  --role pte=qwen3_0_6b_webgpu_q4gsw.pte \
  --role javascript=runner.js \
  --role wasm=runner.wasm

python -m executorch.examples.models.qwen3.webgpu_artifact_manifest validate \
  --root <output-dir> \
  --manifest <output-dir>/manifest.json
```

Validation fails closed on a missing, extra, symlinked, wrong-size or wrong-hash artifact; on a role whose file extension contradicts it; on a method set other than `forward`; on a graph carrying portable operators or a delegate other than the WebGPU one; and on any acquisition pin that disagrees with the checked contract.

### FAQ
For more help with exporting or running this model, feel free to ask in our [discord channel](https://discord.gg/UEjkY9Zs).
