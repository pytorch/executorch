
# Export Llama with OpenVINO Backend

## Download the Model
Follow the [instructions](../../../examples/models/llama/README.md#step-2-prepare-model) to download the required model files. Export Llama with OpenVINO backend is only verified with Llama-3.2-1B variants at this time.

## Environment Setup
Follow the [instructions](../../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

## Export the model:
Navigate into `<executorch_root>/examples/openvino/llama` and execute the commands below to export the model. Update the model file paths to match the location where your model is downloaded. Replace device with the target hardware you want to compile the model for (`CPU`, `GPU`, or `NPU`). The exported model will be generated in the same directory with the filename `llama3_2_ov.pte`. For modifying the output name, change `output_name` in `llama3_2_ov_4wo.yaml` file under `export`.

```
LLAMA_CHECKPOINT=<path/to/model/folder>/consolidated.00.pth
LLAMA_PARAMS=<path/to/model/folder>/params.json
LLAMA_TOKENIZER=<path/to/model/folder>/tokenizer.model

python -m executorch.extension.llm.export.export_llm \
  --config llama3_2_ov_4wo.yaml \
  +backend.openvino.device="CPU" \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}" \
  +base.tokenizer_path="${LLAMA_TOKENIZER:?}"
```

### Compress Model Weights and Export
OpenVINO backend also offers Quantization support for llama models when exporting the model. The different quantization modes that are offered are INT4 groupwise & per-channel weights compression and INT8 per-channel weights compression. It can be achieved by setting `pt2e_quantize` option in `llama3_2_ov_4wo.yaml` file under `quantization`. Set this parameter to `openvino_4wo` for INT4 or `openvino_8wo` for INT8 weight compression. It is set to `openvino_4wo` in `llama3_2_ov_4wo.yaml` file by default. For modifying the group size, set `group_size` option in `llama3_2_ov_4wo.yaml` file under `quantization`. By default group size 128 is used to achieve optimal performance with the NPU.

## Build OpenVINO C++ Runtime with Llama Runner:
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
Update the model tokenizer file path to match the location where your model is downloaded and replace the prompt.
```
./cmake-out/examples/models/llama/llama_main --model_path=<executorch_root>/examples/openvino/llama/llama3_2.pte --tokenizer_path=<path/to/model/folder>/tokenizer.model --prompt="Your custom prompt"
```
