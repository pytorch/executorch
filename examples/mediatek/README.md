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
- Follow the instructions of **Prerequisites** and **Setup** in `backends/mediatek/scripts/README.md`.

## Build MediaTek Examples
1. Build the backend and the examples by exedcuting the script:
```bash
./mtk_build_examples.sh
```

## LLaMa Example Instructions
##### Note: Verify that localhost connection is available before running AoT Flow
1. Exporting Models to `.pte`
- In the `examples/mediatek directory`, run:
```bash
source shell_scripts/export_llama.sh <model_name> <num_chunks> <prompt_num_tokens> <cache_size> <calibration_set_name>
```
- Defaults:
    - `model_name` = llama3
    - `num_chunks` = 4
    - `prompt_num_tokens` = 128
    - `cache_size` = 1024
    - `calibration_set_name` = None
- Argument Explanations/Options:
    - `model_name`: llama2/llama3
    <sub>**Note: Currently Only Tested on Llama2 7B Chat and Llama3 8B Instruct.**</sub>
    - `num_chunks`: Number of chunks to split the model into. Each chunk contains the same number of decoder layers. Will result in `num_chunks` number of `.pte` files being generated. Typical values are 1, 2 and 4.
    - `prompt_num_tokens`: Number of tokens (> 1) consumed each forward pass for the prompt processing stage.
    - `cache_size`: Cache Size.
    - `calibration_set_name`: Name of calibration dataset with extension that is found inside the `aot_utils/llm_utils/prompts` directory. Example: `alpaca.txt`. If `"None"`, will use dummy data to calibrate.
    <sub>**Note: Export script example only tested on `.txt` file.**</sub>

2. `.pte` files will be generated in `examples/mediatek/pte`
    - Users should expect `num_chunks*2` number of pte files (half of them for prompt and half of them for generation).
    - Generation `.pte` files have "`1t`" in their names.
    - Additionally, an embedding bin file will be generated in the weights folder where the `config.json` can be found in. [`examples/mediatek/models/llm_models/weights/<model_name>/embedding_<model_config_folder>_fp32.bin`]
    - eg. For `llama3-8B-instruct`, embedding bin generated in `examples/mediatek/models/llm_models/weights/llama3-8B-instruct/`
    - AoT flow will take roughly 2.5 hours (114GB RAM for `num_chunks=4`) to complete (Results will vary by device/hardware configurations)

### oss
1. Exporting Model to `.pte`
```bash
bash shell_scripts/export_oss.sh <model_name>
```
- Argument Options:
    - `model_name`: deeplabv3/edsr/inceptionv3/inceptionv4/mobilenetv2/mobilenetv3/resnet18/resnet50

# Runtime
## Supported Chips

The examples provided in this repository are tested and supported on the following MediaTek chip:

- MediaTek Dimensity 9300 (D9300)

## Environment Setup

To set up the build environment for the `mtk_executor_runner`:

1. Navigate to the `backends/mediatek/scripts` directory within the repository.
2. Follow the detailed build steps provided in that location.
3. Upon successful completion of the build steps, the `mtk_executor_runner` binary will be generated.

## Deploying and Running on the Device

### Pushing Files to the Device

Transfer the `.pte` model files and the `mtk_executor_runner` binary to your Android device using the following commands:

```bash
adb push mtk_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push <MODEL_NAME>.pte <PHONE_PATH, e.g. /data/local/tmp>
```

Make sure to replace `<MODEL_NAME>` with the actual name of your model file. And, replace the `<PHONE_PATH>` with the desired detination on the device.

##### Note: For oss models, please push additional files to your Android device
```bash
adb push mtk_oss_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push input_list.txt <PHONE_PATH, e.g. /data/local/tmp>
for i in input*bin; do adb push "$i" <PHONE_PATH, e.g. /data/local/tmp>; done;
```

### Executing the Model

Execute the model on your Android device by running:

```bash
adb shell "/data/local/tmp/mtk_executor_runner --model_path /data/local/tmp/<MODEL_NAME>.pte --iteration <ITER_TIMES>"
```

In the command above, replace `<MODEL_NAME>` with the name of your model file and `<ITER_TIMES>` with the desired number of iterations to run the model.

##### Note: For llama models, please use `mtk_llama_executor_runner`. Refer to `examples/mediatek/executor_runner/run_llama3_sample.sh` for reference.
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
