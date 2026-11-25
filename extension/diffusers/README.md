# Diffusers

This directory provides experimental export and Python runtime support for
Hugging Face checkpoints using `StableDiffusionPipeline`,
`StableDiffusionXLPipeline`, or `FluxPipeline`. The exported PTE stores
the text encoder, denoiser and VAE methods together with the metadata needed
by the runner.

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

Use `black-forest-labs/FLUX.1-schnell` or `black-forest-labs/FLUX.1-dev` to
export FLUX.

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

### CUDA export

Export on a Linux machine with a CUDA GPU. The weights are written to a
`.ptd` file next to the PTE:

```bash
python -m executorch.export.huggingface.diffusers.export \
  black-forest-labs/FLUX.1-schnell \
  flux_schnell_cuda.pte \
  --backend cuda \
  --text-encoder-dtype bfloat16 \
  --denoiser-dtype bfloat16 \
  --qlinear 4w
```

`--qlinear` quantizes the denoiser as for MLX; `4w` uses the packed INT4
kernels, which expect a `bfloat16` denoiser, and a group size of 32 unless
`--qlinear-group-size` is given. Pass the `.ptd` file to the runner with
`--data-path`.

### CoreAI export

The experimental CoreAI path lowers text encoders, denoisers and VAEs to
CoreAI. Token IDs and attention masks retain their external INT64 interface;
the lowering pass casts them to INT32 at CoreAI boundaries. It currently
requires static shapes and FP16 or FP32 component precision, and does not
accept `--qlinear`.

Install the pinned SDK packages:

```bash
python -m pip install -r backends/apple/coreai/requirements.txt
```

Export SDXL:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sdxl_coreai.pte \
  --backend coreai \
  --text-encoder-dtype float16 \
  --denoiser-dtype float16 \
  --vae-dtype float32
```

Running this PTE requires macOS 27 or newer and an ExecuTorch runtime built
with `EXECUTORCH_BUILD_COREAI=ON`. Python execution
also requires `EXECUTORCH_BUILD_PYBIND=ON`. Build against an Apple SDK 27 or
newer. See
[the CoreAI backend README](../../backends/apple/coreai/README.md) for runtime
requirements. Initial loading may specialize the CoreAI models and requires
additional disk space for model assets and caches.
