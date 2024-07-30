# Summary

## Overview
This file provides you the instructions to run LLAMA2 with different parameters via Qualcomm HTP backend. Following settings support for Llama-2-7b-chat-hf

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
python examples/qualcomm/qaihub_scripts/llama2/qaihub_llama2_7b.py -a ${ARTIFACTS} -b build_android -s ${SERIAL_NUM} -m ${SOC_MODEL} --context_binaries ${AIHUB_CONTEXT_BINARIES} --tokenizer_bin tokenizer.bin --prompt "What is Python?"
```