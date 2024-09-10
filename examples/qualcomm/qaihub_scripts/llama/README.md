# Summary

## Overview
This file provides you the instructions to run LLAMA2 and LLAMA3 with different parameters via Qualcomm HTP backend. Following settings support for Llama-2-7b-chat-hf and Llama-3-8b-chat-hf

Please check corresponding section for more information.

## Llama-2-7b-chat-hf
This example demonstrates how to run Llama-2-7b-chat-hf on mobile via Qualcomm HTP backend. Model was precompiled into context binaries by [Qualcomm AI HUB](https://aihub.qualcomm.com/).
Note that the pre-compiled context binaries could not be futher fine-tuned for other downstream tasks.

### Instructions
#### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

#### Step2: Prepare Model
1. Create account for https://aihub.qualcomm.com/
2. Follow instructions in https://huggingface.co/qualcomm/Llama-v2-7B-Chat to export context binaries (will take some time to finish)

```bash
# tokenizer.model: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer.model
# tokenizer.bin:
python -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
```

#### Step3: Run default examples
```bash
# AIHUB_CONTEXT_BINARIES: ${PATH_TO_AIHUB_WORKSPACE}/build/llama_v2_7b_chat_quantized
python examples/qualcomm/qaihub_scripts/llama/llama2/qaihub_llama2_7b.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --context_binaries ${AIHUB_CONTEXT_BINARIES} --tokenizer_bin tokenizer.bin --prompt "What is Python?"
```

## Llama-3-8b-chat-hf
This example demonstrates how to run Llama-3-8b-chat-hf on mobile via Qualcomm HTP backend. Model was precompiled into context binaries by [Qualcomm AI HUB](https://aihub.qualcomm.com/).
Note that the pre-compiled context binaries could not be futher fine-tuned for other downstream tasks. This example script has been tested on a 16GB RAM device and verified to work.

### Instructions
#### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

#### Step2: Prepare Model
1. Create account for https://aihub.qualcomm.com/
2. Follow instructions in https://huggingface.co/qualcomm/Llama-v3-8B-Chat to export context binaries (will take some time to finish)
3. For Llama 3 tokenizer, please refer to https://github.com/meta-llama/llama-models/blob/main/README.md for further instructions on how to download tokenizer.model.


#### Step3: Run default examples
```bash
# AIHUB_CONTEXT_BINARIES: ${PATH_TO_AIHUB_WORKSPACE}/build/llama_v3_8b_chat_quantized
python examples/qualcomm/qaihub_scripts/llama/llama3/qaihub_llama3_8b.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --context_binaries ${AIHUB_CONTEXT_BINARIES} --tokenizer_model tokenizer.model --prompt "What is baseball?"
```