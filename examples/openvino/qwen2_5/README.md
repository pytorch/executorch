
# Export Qwen2.5-1.5B with OpenVINO Backend

## Download the Model

Download the Qwen2.5-1.5B checkpoint from HuggingFace:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir <path/to/model/folder>
```

Then convert the HuggingFace safetensors checkpoint to Meta format:

```bash
python examples/models/qwen2_5/convert_weights.py <path/to/model/folder> <path/to/output/consolidated.00.pth>
```

## Environment Setup

Follow the [instructions](../../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

## Export the Model

Execute the commands below from `<executorch_root>`. Update the model file paths to match the location where your model is downloaded. Replace device with the target hardware you want to compile the model for (`CPU`). The exported model will be generated in the current directory with the filename `qwen2_5_1_5b_ov.pte`. To modify the output name, change `output_name` in `examples/openvino/qwen2_5/qwen2_5_1_5b_ov_4wo.yaml` under `export`.

```bash
QWEN_CHECKPOINT=<path/to/consolidated.00.pth>

python -m executorch.extension.llm.export.export_llm \
  --config examples/openvino/qwen2_5/qwen2_5_1_5b_ov_4wo.yaml \
  +backend.openvino.device="CPU" \
  +base.model_class="qwen2_5_1_5b" \
  +base.checkpoint="${QWEN_CHECKPOINT:?}" \
  +base.params="examples/models/qwen2_5/config/1_5b_config.json"
```

### Compress Model Weights and Export

OpenVINO backend also offers quantization support for weight compression. The different quantization modes available are INT4 groupwise & per-channel weights compression and INT8 per-channel weights compression. Set `pt2e_quantize` in `examples/openvino/qwen2_5/qwen2_5_1_5b_ov_4wo.yaml` under `quantization` to `openvino_4wo` for INT4 or `openvino_8wo` for INT8 weight compression. It is set to `openvino_4wo` by default. To modify the group size, set `group_size` under `quantization`. By default group size 128 is used to achieve optimal performance with the NPU.

## Build OpenVINO C++ Runtime with Llama Runner

First, build the backend libraries with llm extension by executing the script below in `<executorch_root>/backends/openvino/scripts` folder:

```bash
./openvino_build.sh --cpp_runtime_llm
```

Then, build the llama runner by executing commands below in `<executorch_root>` folder:

```bash
# Configure the project with CMake
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -Bcmake-out/examples/models/llama \
      examples/models/llama
# Build the llama runner
cmake --build cmake-out/examples/models/llama -j$(nproc) --config Release
```

The executable is saved in `<executorch_root>/cmake-out/examples/models/llama/llama_main`

## Execute Inference Using Llama Runner

Qwen2.5 uses a HuggingFace tokenizer. Update the tokenizer path to match the location where your model is downloaded and replace the prompt.

```bash
./cmake-out/examples/models/llama/llama_main \
  --model_path=<executorch_root>/examples/openvino/qwen2_5/qwen2_5_1_5b_ov.pte \
  --tokenizer_path=<path/to/model/folder>/tokenizer.json \
  --prompt="Your custom prompt"
```
