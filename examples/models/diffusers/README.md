# Diffusers C++ runner

`diffusion_runner` generates images from diffusion programs exported with
`export/huggingface/diffusers`. It supports the `sd15`, `sdxl` and `flux`
pipelines and is the C++ counterpart of
`python -m executorch.extension.diffusers.runner`.

## Build

```bash
make diffusers-mlx
```

The binary is written to `cmake-out/examples/models/diffusers/diffusion_runner`.

## Run

The runner reads the tokenizers and the scheduler configuration from a
Hugging Face snapshot of the exported model. Download them into a local
directory. CLIP tokenizers ship as `vocab.json` and `merges.txt`, which the
runner does not read, so convert each CLIP tokenizer folder to
`tokenizer.json` once:

```bash
MODEL_DIR=flux1-schnell-assets
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('black-forest-labs/FLUX.1-schnell', local_dir='$MODEL_DIR', \
allow_patterns=['tokenizer*/*', 'scheduler/*'])"
python -c "from transformers import CLIPTokenizerFast as T; \
T.from_pretrained('$MODEL_DIR/tokenizer').save_pretrained('$MODEL_DIR/tokenizer')"

cmake-out/examples/models/diffusers/diffusion_runner \
  --model_path flux_schnell_mlx.pte \
  --model_dir "$MODEL_DIR" \
  --prompt "a photo of a cat" \
  --seed 9 \
  --output cat.png
```

FLUX.1 uses T5 in `tokenizer_2`, which loads from `spiece.model` as is. SDXL
has a second CLIP tokenizer in `tokenizer_2`, which needs the same conversion.

Unset options such as `--steps`, `--cfg_scale`, `--guidance_scale`,
`--height` and `--width` use the defaults recorded in the program. A seed
produces the same initial noise as the Python runner.

`--image input.png --strength 0.8` starts from an image instead of noise
(Stable Diffusion programs, which export an `encode` method). The image size
is the output size unless `--height` and `--width` are given, and it must be
a size the program accepts.

`--preview_every N` decodes the current latents every `N` steps and writes
them to `--output`, so an image viewer showing that file follows the
progress. Each preview costs one extra VAE decode.
