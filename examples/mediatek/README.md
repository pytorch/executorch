# Directory Structure

Below is the layout of the `examples/mediatek` directory, which includes the necessary files for the example applications:

```plaintext
examples/mediatek
├── aot_utils                         # Utils for AoT export
    ├── llm_utils                     # Utils for LLM models
        ├── preformatter_templates    # Model specific prompt preformatter templates
        ├── tokenizers_               # Model tokenizer scripts
├── model_export_scripts              # Model specifc export scripts
├── models                            # Model definitions
    ├── llm_models                    # LLM model definitions
        ├── weights                   # LLM model weights location (Offline) [Ensure that config.json, relevant tokenizer files and .bin or .safetensors weights file(s) are placed here]
├── executor_runner                   # Example C++ wrapper for the ExecuTorch runtime
├── pte                               # Generated .pte files location
├── shell_scripts                     # Shell scripts to quickrun model specific exports
├── CMakeLists.txt                    # CMake build configuration file for compiling examples
├── requirements.txt                  # MTK and other required packages
└── README.md                         # Documentation for the examples (this file)
```
# AoT
## Environment Setup
In addition to the Executorch environment setup, refer to the `requirements.txt` file.
## AoT Flow
1. Exporting Models to `.pte`
- In the `examples/mediatek directory`, run:
```bash
source shell_scripts/export_llama.sh <model_name> <num_chunks> <prompt_num_tokens> <cache_size>
```
- Defaults:
    - `model_name` = llama3
    - `num_chunks` = 4
    - `prompt_num_tokens` = 128
    - `cache_size` = 1024
- Argument Explanations/Options:
    - `model_name`: llama2/llama3
    <sub>**Note: Currently Only Tested on Llama2 7B Chat and Llama3 8B Instruct.**</sub>
    - `num_chunks`: Number of chunks to split the model into. Each chunk contains the same number of decoder layers. Will result in `num_chunks` number of `.pte` files being generated. Typical values are 1, 2 and 4.
    - `prompt_num_tokens`: Number of tokens (> 1) consumed each forward pass for the prompt processing stage.
    - `cache_size`: Cache Size.

2. `.pte` files will be generated in `examples/mediatek/pte`

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

### Executing the Model

Execute the model on your Android device by running:

```bash
adb shell "/data/local/tmp/mtk_executor_runner --model_path /data/local/tmp/<MODEL_NAME>.pte --iteration <ITER_TIMES>"
```

In the command above, replace `<MODEL_NAME>` with the name of your model file and `<ITER_TIMES>` with the desired number of iterations to run the model.
