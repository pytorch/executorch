# Summary

## Overview
This file provides you the instructions to run Stable-Diffusion-v2.1 with different parameters via Qualcomm HTP backend. We will demonstrate how to run Stable Diffusion v2.1 on mobile devices using context binaries from Qualcomm AI Hub’s Stable Diffusion v2.1

Please check corresponding section for more information.

## Stable-Diffusion-v2.1
The model architecture, scheduler, and time embedding are from the [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

### Instructions
#### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

#### Step2: Prepare Model
1. Download the context binaries for TextEncoder, UNet, and VAEDecoder under https://huggingface.co/qualcomm/Stable-Diffusion-v2.1/tree/main
2. Download vocab.json under https://huggingface.co/openai/clip-vit-base-patch32/tree/main


#### Step3: Install Requirements
Before running the code, you need to install the necessary Python packages.

We have verified the code with `diffusers`==0.29.0 and `piq`==0.8.0. Please follow the instructions here to install the required items:
```bash
sh examples/qualcomm/qaihub_scripts/stable_diffusion/install_requirements.sh
```

#### Step4: Run default example
In this example, we execute the script for 20 time steps with the `prompt` 'a photo of an astronaut riding a horse on mars':
```bash
python examples/qualcomm/qaihub_scripts/stable_diffusion/qaihub_stable_diffusion.py -b build-android -m ${SOC_MODEL} --s ${SERIAL_NUM} --text_encoder_bin ${PATH_TO_TEXT_ENCODER_CONTEXT_BINARY} --unet_bin ${PATH_TO_UNET_CONTEXT_BINARY} --vae_bin ${PATH_TO_VAE_CONTEXT_BINARY} --vocab_json  ${PATH_TO_VOCAB_JSON_FILE} --num_time_steps 20 --prompt "a photo of an astronaut riding a horse on mars"
```
- Please replace `${PATH_TO_TEXT_ENCODER_CONTEXT_BINARY}`, `${PATH_TO_UNET_CONTEXT_BINARY}`, and `${PATH_TO_VAE_CONTEXT_BINARY}` with the actual paths to your AI Hub context binary files.
- Please replace `${PATH_TO_VOCAB_JSON_FILE}` with the actual path to your vocab.json file.
