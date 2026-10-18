# Diffusers

This directory provides experimental export and Python runtime support for
Hugging Face checkpoints using `StableDiffusionPipeline` or
`StableDiffusionXLPipeline`. The exported PTE stores the text encoder, denoiser
and VAE methods together with the metadata needed by the runner.

All Python APIs in this package are experimental and may change without notice.

## Installation

Install the optional diffusion dependencies:

```bash
python -m pip install -r extension/diffusers/requirements.txt
```

## Usage

### Export

Run from the ExecuTorch repository root:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stable-diffusion-v1-5/stable-diffusion-v1-5 \
  stable_diffusion_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16
```

Use `stabilityai/stable-diffusion-xl-base-1.0` to export SDXL.

### Run

```bash
python -m executorch.extension.diffusers.runner \
  stable_diffusion_mlx.pte \
  --model-id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --prompt "a photo of a cat" \
  --steps 50 \
  --seed 9 \
  --output cat.png
```

The runner downloads the matching Hugging Face tokenizers and scheduler
specified by `--model-id`.

## Configuration options

### Dynamic shapes

Export SDXL with dynamic spatial dimensions:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sdxl_dynamic_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16 \
  --vae-dtype float32 \
  --dynamic-shapes \
  --min-size 512 \
  --max-size 1024
```

Run it at a resolution within that range:

```bash
python -m executorch.extension.diffusers.runner \
  sdxl_dynamic_mlx.pte \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "an astronaut riding a horse" \
  --height 768 \
  --width 1024 \
  --output astronaut.png
```

### Quantization

Quantize the denoiser's linear weights during MLX export:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sdxl_4w_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16 \
  --vae-dtype float32 \
  --qlinear 4w \
  --qlinear-group-size 64
```

MLX linear quantization supports `4w`, `8w` and `nvfp4`; supported group sizes
are 32, 64 and 128.
