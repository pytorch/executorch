# Summary

## Overview
This file provides instructions to run LLAMA3.2 1B and 3B (WIP) with different parameters via the Qualcomm HTP backend. In LLAMA3.2, we offer the following modes to execute the model:

Prefill Mode: This is also known as batch prefill mode, where the model takes in a list of tokens as input and generates the next token along with the key-value (KV) cache for all tokens. This mode is efficient for generating the initial sequence of tokens (usually the user's prompt).

KV Cache Mode: In KV Cache mode, the model takes in a single previous token and generates the next predicted token along with its KV cache. It is efficient for generating subsequent tokens after the initial prompt.

Hybrid Mode: Hybrid mode leverages the strengths of both batch prefill and KV cache modes to optimize token generation speed. Initially, it uses prefill mode to efficiently generate the prompt's key-value (KV) cache. Then, the mode switches to KV cache mode, which excels at generating subsequent tokens.

## Instructions
### Note
1. For hybrid mode, the export time will be longer and can take up to 2-4 hours to complete.
2. When exporting a hybrid mode model, please ensure the device has at least 80 GB of memory and swap space.

### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

### Step 2: Prepare Model
1. Follow the [instructions](https://www.llama.com/) to download models.
At the end of this step, users should have the following files ready: consolidated.00.pth, params.json, and tokenizer.model.

### Step3: Run default examples using hybrid mode.
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama3_2/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --prompt "what is 1+1" --temperature 0 --model_size 1B --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128
```

If you would like to compile the model only, we have provided the flag `--compile_only`.
```bash
python examples/qualcomm/oss_scripts/llama3_2/llama.py -b build-android -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --prompt "what is 1+1" --temperature 0 --model_size 1B --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --compile_only
```

On the other hand, if you already have a pre-compiled .pte model, you can perform inference by providing the flag `--pre_gen_pte` and specifying the folder that contains the .pte model.
```bash
python examples/qualcomm/oss_scripts/llama3_2/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --prompt "what is 1+1" --temperature 0 --model_size 1B --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```