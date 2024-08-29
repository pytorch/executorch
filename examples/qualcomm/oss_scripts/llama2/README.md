# Summary

## Overview
This file provides you the instructions to run LLAMA2 with different parameters via Qualcomm HTP backend. Following settings support for Stories 110M

Please check corresponding section for more information.

## Stories 110M
This example demonstrates how to run a smaller LLAMA2, stories110M on mobile via Qualcomm HTP backend. Model architecture is fine-tuned specifically for HTP to accelerate the performance. Weight is quantized via PTQ quantization to fit the model on a phone.

### Instructions
#### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

#### Step2: Prepare Model
Download and preapre stories110M model

```bash
# tokenizer.model & stories110M.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"

# tokenizer.bin:
python -m extension.llm.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

#### Step3: Run default examples
Default example generates the story based on the given prompt, "Once".
```bash
# 16a4w quant:
python examples/qualcomm/oss_scripts/llama2/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint stories110M --params params.json --tokenizer_model tokenizer.model --tokenizer_bin tokenizer.bin --prompt "Once"
```

#### (Note) Customized PTQ data set
User prompts are used for PTQ calibration data. Take the examples above, the word "Once" is the only word for PTQ. If you want to observe more data during the calibration time. Please add more prompts to the args `--prompt`.