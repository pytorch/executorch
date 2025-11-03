# Stable Diffusion LCM with OpenVINO Backend

This example demonstrates how to run Latent Consistency Models (LCM) for fast text-to-image generation on Intel hardware using ExecuTorch with the OpenVINO backend.

## Overview

Latent Consistency Models (LCMs) are optimized diffusion models that generate high-quality images in just 4-8 steps, compared to 25-50 steps required by traditional Stable Diffusion models.

## Environment Setup
Follow the [instructions](../../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

### Install dependencies
```bash
pip install -r requirements.txt
```

## Export the Model

Export the LCM model:

```bash
python export_lcm.py \
    --model_id SimianLuo/LCM_Dreamshaper_v7 \
    --output_dir ./lcm_models \
    --device CPU \
    --dtype fp16
```
This will create three files in `./lcm_models/`:
- `text_encoder.pte`
- `unet.pte`
- `vae_decoder.pte`

### Generate Images

Run inference with the exported model:

```bash
python openvino_lcm.py \
    --models_dir ./lcm_models \
    --prompt "a beautiful sunset over mountains" \
    --steps 4 \
    --dtype fp16
```
## Supported Models

This implementation supports LCM-based Stable Diffusion models:
- **SimianLuo/LCM_Dreamshaper_v7**
- **latent-consistency/lcm-sdxl**
