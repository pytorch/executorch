# Directory Structure

Below is the layout of the `examples/mediatek` directory, which includes the necessary files for the example applications:

```plaintext
examples/mediatek
├── aot_utils                         # Utils for AoT export
    ├── llm_utils                     # Utils for LLM models
        ├── preformatter_templates    # Model specific prompt preformatter templates
        ├── prompts                   # Calibration Prompts
        ├── tokenizers_               # Model tokenizer scripts
    ├── oss_utils                     # Utils for oss models
├── eval_utils                        # Utils for eval oss models
├── model_export_scripts              # Model specifc export scripts
├── models                            # Model definitions
    ├── llm_models                    # LLM model definitions
        ├── weights                   # LLM model weights location (Offline) [Ensure that config.json, relevant tokenizer files and .bin or .safetensors weights file(s) are placed here]
├── executor_runner                   # Example C++ wrapper for the ExecuTorch runtime
├── pte                               # Generated .pte files location
├── shell_scripts                     # Shell scripts to quickrun model specific exports
├── CMakeLists.txt                    # CMake build configuration file for compiling examples
├── requirements.txt                  # MTK and other required packages
├── mtk_build_examples.sh             # Script for building MediaTek backend and the examples
└── README.md                         # Documentation for the examples (this file)
```
# Examples Build Instructions

## Environment Setup
- Follow the instructions in `backends/mediatek/README.md` to build the backend library `libneuron_backend.so`.

## Build MediaTek Runners
1. Build the mediatek model runner by executing the script:
```bash
./mtk_build_examples.sh
```
This will generate the required runners in `executorch/cmake-android-out/examples/mediatek/`

## Model Export Instructions
##### Note: Verify that localhost connection is available before running AoT Flow
1. Download Required Files
- Download the model files from the official Hugging Face website, and move the files to the respective folder in `examples/mediatek/models/llm_models/weights/` **EXCEPT** the `config.json` file.
    - The `config.json` file is already included in the model folders, which may include some modifications required for the model exportation.
- Include the calibration data (if any) under `aot_utils/llm_utils/prompts/`

2. Exporting Models to `.pte`
- In the `examples/mediatek/ directory`, run:
```bash
source shell_scripts/export_<model_family>.sh <model_name> <num_chunks> <prompt_num_tokens> <cache_size> <calibration_data_file> <precision> <platform>
```
- Defaults:
    - `model_name` = Depends on model family. Check respective `shell_scripts/export_<model_family>.sh` for info.
    - `num_chunks` = 4
    - `prompt_num_tokens` = 128
    - `cache_size` = 512
    - `calibration_data_file` = None
    - `precision` = A16W4
    - `platform` = DX4

- Argument Explanations/Options:
    - `model_name`: View list 'Available model names' below.
    - `num_chunks`: Number of chunks to split the model into. Each chunk contains the same number of decoder layers. Typical values are 1, 2 and 4.
    - `prompt_num_tokens`: Number of tokens (> 1) consumed each forward pass for the prompt processing stage.
    - `cache_size`: Cache Size.
    - `calibration_data_file`: Name of calibration dataset with extension that is found inside the `aot_utils/llm_utils/prompts/` directory. Example: `alpaca.txt`. If `"None"`, will use dummy data to calibrate.
    - `precision`: Quantization precision for the model. Available options are `["A16W4", "A16W8", "A16W16", "A8W4", "A8W8"]`
    - `platform`: The platform of the device. `DX4` for Mediatek Dimensity 9400 and `DX3` for Mediatek Dimensity 9300.
    <sub>**Note: Export script example only tested on `.txt` file.**</sub>

- Available model names:
    - Llama:
        - llama3.2-3b, llama3.2-1b, llama3, llama2
    - Qwen:
        - Qwen3-4B, Qwen3-1.7B, Qwen2-7B-Instruct, Qwen2.5-3B, Qwen2.5-0.5B-Instruct, Qwen2-1.5B-Instruct
    - Gemma:
        - gemma2, gemma3
    - Phi:
        - phi3.5, phi4

3. `.pte` files will be generated in `examples/mediatek/pte/`
    - Users should expect `num_chunks` number of pte files.
    - An embedding bin file will be generated in the weights folder where the `config.json` can be found in. [`examples/mediatek/models/llm_models/weights/<model_name>/embedding_<model_config_folder>_fp32.bin`]
    - eg. For `llama3-8B-instruct`, embedding bin generated in `examples/mediatek/models/llm_models/weights/llama3-8B-instruct/`
    - AoT flow will take around 30 minutes to 2.5 hours to complete (Results will vary depending on device/hardware configurations and model sizes)

### oss
1. Exporting Model to `.pte`
```bash
bash shell_scripts/export_oss.sh <model_name>
```
- Argument Options:
    - `model_name`: deeplabv3/edsr/inceptionv3/inceptionv4/mobilenetv2/mobilenetv3/resnet18/resnet50/dcgan/wav2letter/vit_b_16/mobilebert/emformer_rnnt/bert/distilbert

# Runtime
## Deploying and Running on the Device

### Pushing Files to the Device

Transfer the directory containing the `.pte` model files, the `run_<model_name>_sample.sh` script, the `embedding_<model_config_folder>_fp32.bin`, the tokenizer file, the `mtk_llama_executor_runner` binary and the 3 `.so` files to your Android device using the following commands:

```bash
adb push mtk_llama_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push examples/mediatek/executor_runner/run_<model_name>_sample.sh <PHONE_PATH, e.g. /data/local/tmp>
adb push embedding_<model_config_folder>_fp32.bin <PHONE_PATH, e.g. /data/local/tmp>
adb push tokenizer.model <PHONE_PATH, e.g. /data/local/tmp>
adb push <PTE_DIR> <PHONE_PATH, e.g. /data/local/tmp>
```

Make sure to replace `<PTE_DIR>` with the actual name of your directory containing pte files. And, replace the `<PHONE_PATH>` with the desired detination on the device.

At this point your phone directory should have the following files:
- libneuron_backend.so
- libneuronusdk_adapter.mtk.so
- libneuron_buffer_allocator.so
- mtk_llama_executor_runner
- <PTE_DIR>
- tokenizer.json / tokenizer.model(for llama3) / tokenizer.bin(for phi3 and gemma2)
- embedding_<model_config_folder>_fp32.bin
- run_<model_name>_sample.sh

##### Note: For oss models, please push additional files to your Android device
```bash
adb push mtk_oss_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push input_list.txt <PHONE_PATH, e.g. /data/local/tmp>
for i in input*bin; do adb push "$i" <PHONE_PATH, e.g. /data/local/tmp>; done;
```

### Executing the Model

Execute the model on your Android device by running:

```bash
adb shell
cd <PHONE_PATH>
sh run_<model_name>_sample.sh
```
#### Note: The `mtk_llama_executor_runner` is applicable to the models listed in `examples/mediatek/models/llm_models/weights/`.

##### Note: For non-LLM models, please run `adb shell "/data/local/tmp/mtk_executor_runner --model_path /data/local/tmp/<MODEL_NAME>.pte --iteration <ITER_TIMES>"`. 
##### Note: For oss models, please use `mtk_oss_executor_runner`.
```bash
adb shell "/data/local/tmp/mtk_oss_executor_runner --model_path /data/local/tmp/<MODEL_NAME>.pte --input_list /data/local/tmp/input_list.txt --output_folder /data/local/tmp/output_<MODEL_NAME>"
adb pull "/data/local/tmp/output_<MODEL_NAME> ./"
```

### Check oss result on PC
```bash
python3 eval_utils/eval_oss_result.py --eval_type <eval_type> --target_f <golden_folder> --output_f <prediction_folder>
```
For example:
```
python3 eval_utils/eval_oss_result.py --eval_type piq --target_f edsr --output_f output_edsr
```
- Argument Options:
    - `eval_type`: topk/piq/segmentation
    - `target_f`: folder contain golden data files. file name is `golden_<data_idx>_0.bin`
    - `output_f`: folder contain model output data files. file name is `output_<data_idx>_0.bin`
