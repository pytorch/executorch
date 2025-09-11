
# Export Llama with OpenVINO Backend

## Download the Model
Follow the [instructions](../../examples/models/llama#step-2-prepare-model) to download the required model files. Export Llama with OpenVINO backend is only verified with Llama-3.2-1B variants at this time. 

## Environment Setup
Follow the [instructions](../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

## Export the model:
Navigate into `<executorch_root>/examples/openvino/llama` and execute the commands below to export the model. Update the model file paths to match the location where your model is downloaded. The exported model will be generated in the same directory with the filename `llama3_2.pte`.

```
LLAMA_CHECKPOINT=<path/to/model/folder>/consolidated.00.pth
LLAMA_PARAMS=<path/to/model/folder>/params.json
LLAMA_TOKENIZER=<path/to/model/folder>/tokenizer.model

python -m executorch.extension.llm.export.export_llm \
  --config llama3_2_ov_4wo_config.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}" \
  +base.tokenizer_path="${LLAMA_TOKENIZER:?}"
```

## Build OpenVINO C++ Runtime with Llama Runner:
First, build the backend libraries by executing the script below in `<executorch_root>/backends/openvino/scripts` folder:
```bash
./openvino_build.sh --cpp_runtime
```
Then, build the llama runner by executing the script below (with `--llama_runner` argument) also in `<executorch_root>/backends/openvino/scripts` folder:
```bash
./openvino_build.sh --llama_runner
```
The executable is saved in `<executorch_root>/cmake-out/examples/models/llama/llama_main`

## Execute Inference Using Llama Runner
Update the model tokenizer file path to match the location where your model is downloaded and replace the prompt.
```
./cmake-out/examples/models/llama/llama_main --model_path=<executorch_root>/examples/openvino/llama/llama3_2.pte --tokenizer_path=<path/to/model/folder>/tokenizer.model --prompt="Your custom prompt"
```
