# SmolLM2 → VGF Quickstart

> **Heads-up:** Use FP32 as the accuracy baseline and `linear16a8w` with an
> FP32 KV cache as the recommended quantized path. Treat static INT8 KV-cache
> storage and `linear8a8w` as experimental.

This is a host-only VGF workflow built around `executor_runner`. Run the
commands from the root of an ExecuTorch source checkout.

## 0. Prerequisites
Run all commands from the repository root.

Install the Arm MLSDK/VKML dependencies and generate `setup_path.sh`:

```bash
examples/arm/setup.sh \
  --i-agree-to-the-contained-eula \
  --disable-ethos-u-deps \
  --enable-mlsdk-deps \
  --enable-emulation-layer
```

Activate your Python environment and source the generated Arm setup:

```bash
# Python env (example)
source env/bin/activate

# Arm tools + VKML emulation
source examples/arm/arm-scratch/setup_path.sh
```

If you want the broader Arm backend setup flow, see
`examples/arm/README.md`. This README only covers the SmolLM2 VGF host path.

## 1. Tokenizer (one-time)
```bash
mkdir -p data/tokenizers/smollm2
huggingface-cli download HuggingFaceTB/SmolLM2-135M tokenizer.json \
  --local-dir data/tokenizers/smollm2
```
The download lives at `data/tokenizers/smollm2/tokenizer.json`. Use this path in the export and sampling commands below.

If you see CMake complaining that your GCC is “too new” for CUDA when building
the VKML runner, use a CUDA-supported host compiler, e.g.:

```bash
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=$CXX
```

## 2. Baseline: full-FP32 model and KV cache
Produces a stable `.pte` for experimentation and sampling. KV cache is the
preferred runtime flow because each decode step reuses cached attention state
instead of recomputing the full prompt window.

```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_kv_fp32 \
  export.output_name=smollm2_vgf_kv_fp32.pte \
  export.max_seq_length=128 \
  export.max_context_length=128 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP+INT \
  model.use_kv_cache=True \
  model.enable_dynamic_shape=False \
  debug.verbose=True
```

## 3. Recommended quantized path: `linear16a8w` with FP32 KV storage
This quantizes `torch.nn.Linear` modules with the Arm VGF PT2E quantizer and
keeps the persistent KV cache in FP32. This is the recommended starting point
because static INT8 cache storage has not yet demonstrated a performance
benefit on VGF hardware.

Example (16-bit activations, 8-bit weights, Linear-only):

```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_kv_linear16a8w \
  export.output_name=smollm2_vgf_kv_linear16a8w.pte \
  export.max_seq_length=128 \
  export.max_context_length=128 \
  quantization.pt2e_quantize=vgf_16a8w \
  quantization.calibration_tasks=\[wikitext\] \
  quantization.calibration_limit=64 \
  quantization.calibration_seq_length=128 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP+INT+int16 \
  backend.vgf.quantize_scope=linear \
  model.use_kv_cache=True \
  model.enable_dynamic_shape=False \
  debug.verbose=True
```

### 3.1 Experimental: `linear16a8w` with static INT8 KV storage

This path stores the persistent KV cache as calibrated INT8 data. It is
different from `model.quantize_kv_cache=True`, which performs dynamic per-token
KV quantization and currently leaves cache QDQ/update maintenance outside the
VGF delegate.

When `quantization.calibration_tasks` is set, export first runs the task
through a float KV cache and derives separate symmetric per-head-dimension K/V
scales for every layer. The same task is then reused for normal PT2E Linear
calibration. Without a calibration task, static KV storage uses
`model.static_quantize_kv_cache_scale` as a fixed fallback;
`calibration_data` alone only calibrates the PT2E operators.

Use the recommended command above with these changes:

```bash
export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_static_kvq_linear16a8w \
export.output_name=smollm2_vgf_static_kvq_linear16a8w.pte \
model.static_quantize_kv_cache=True
```

The static KV-cache path uses symmetric per-head-dimension scales with
zero-point `0` and dequantizes the cache back to float for the existing
attention path. Treat its accuracy and performance as experimental until they
have been validated for the target hardware and context length.

Alternative `linear8a8w` mode:

```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_kv_linear8a8w \
  export.output_name=smollm2_vgf_kv_linear8a8w.pte \
  export.max_seq_length=128 \
  export.max_context_length=128 \
  quantization.pt2e_quantize=vgf_8a8w \
  quantization.calibration_tasks=\[wikitext\] \
  quantization.calibration_limit=64 \
  quantization.calibration_seq_length=128 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP+INT \
  backend.vgf.quantize_scope=linear \
  model.use_kv_cache=True \
  model.enable_dynamic_shape=False \
  debug.verbose=True
```

`quantization.pt2e_quantize` selects the numeric mode.
`backend.vgf.quantize_scope=linear` keeps quantization limited to
`torch.nn.Linear` modules. The compile spec still includes FP because the rest
of the graph remains floating point.

### 3.2 Alternative: static non-KV full-logits exports
Use the static non-KV path when you specifically need full-window logits, for
example faster perplexity evaluation or debugging padded fixed-shape behavior.
For calibrated non-KV exports, keep `debug.generate_full_logits=True`; this lets
the helpers select the last real-token logits row instead of the padded tail.
Set `model.use_kv_cache=False`, keep `export.max_seq_length` and
`quantization.calibration_seq_length` aligned, and add
`debug.generate_full_logits=True` to the export commands above.

## 4. Sampling with `executor_runner`

### 4.0 Build `executor_runner` (VKML)
```bash
source examples/arm/arm-scratch/setup_path.sh

rm -rf cmake-out-vkml
bash examples/arm/smollm2_example_vgf/build_executor_runner_vkml.sh cmake-out-vkml
```

This example-specific wrapper enables `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`
in addition to the VGF and quantized kernel flags. That matters for the SmolLM2
FP32 path, where the generic VKML build helper may not provide enough fallback
CPU kernel coverage.

### 4.1 Greedy and `T=0.8` sampling
`examples/arm/smollm2_example_vgf/generate_sampled.py` wraps
`cmake-out-vkml/executor_runner` and supports both KV-cache and static non-KV
exports. The recommended KV-cache path uses `--use-kv-cache`; the static
non-KV path uses a fixed token window plus `--full-logits`.

Greedy generation (`--temperature 0`) always chooses the highest-logit next
token, which is useful for deterministic comparisons. Stochastic generation
(`--temperature 0.8` with `--top-p 0.9`) samples from a filtered probability
distribution, so it can produce more varied text while still being reproducible
with a fixed `--seed`.

Notes:
- `--max-seq-length` must match the export `export.max_seq_length` (otherwise you will hit input size mismatch).
- KV-cache exports take `token,input_pos` inputs and should pass `--use-kv-cache`.
- Static non-KV exports take an `int32[1, max_seq_length]` token window and should pass `--full-logits`.
- The KV-cache path keeps `executor_runner` alive in server mode automatically.
- The documented examples use `--temperature 0` (greedy) and `--temperature 0.8`.
- For deterministic comparisons against saved `temp0` outputs, use `--seed 0`, `--repetition-penalty 1.1`, and `--no-topk-print`. At `--temperature 0`, token selection is greedy, so `--top-p` does not affect the chosen token.

Greedy example (`T=0`):
```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_kv_linear16a8w.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --max-seq-length 128 \
  --max-context-length 128 \
  --max-new-tokens 10 \
  --seed 0 \
  --temperature 0 \
  --repetition-penalty 1.1 \
  --use-kv-cache
```

Stochastic example (`T=0.8`):
```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_kv_linear16a8w.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --max-seq-length 128 \
  --max-context-length 128 \
  --max-new-tokens 10 \
  --seed 0 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --use-kv-cache
```
> Swap `--pte` to `smollm2_vgf_static_kvq_linear16a8w.pte` to compare
> experimental static INT8 KV storage, or to `smollm2_vgf_kv_fp32.pte` for the
> full-FP32 baseline.



### 4.2 Batch prompts from `default_prompts.txt`

To generate for *all* prompts in `default_prompts.txt` and save to a file:

```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_kv_linear16a8w.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt-file examples/arm/smollm2_example_vgf/default_prompts.txt \
  --prompt-all \
  --max-seq-length 128 \
  --max-context-length 128 \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --use-kv-cache \
  --save-generations outputs/$(date +%F)/$(date +%H-%M-%S)_smollm2_gen.txt
```

## 5. Wikitext prompts and perplexity

For a smoke test before a dedicated comparison, run the `default_prompts.txt`
command above with `smollm2_vgf_kv_linear16a8w.pte` and inspect the
generations. To evaluate the experimental static INT8 cache artifact alone,
pass `smollm2_vgf_static_kvq_linear16a8w.pte` to any one of the PTE options
below.

Build a reusable 1000-prompt file from `wikitext-2-raw-v1` and evaluate
perplexity on the first 100 prompts for KV-cache FP32, `linear8a8w`, and
`linear16a8w`:

```bash
OUT_DIR=outputs/$(date +%F)/$(date +%H-%M-%S)_smollm2_vgf_eval

python examples/arm/smollm2_example_vgf/eval_wikitext_perplexity.py \
  --runner cmake-out-vkml/executor_runner \
  --pte-fp32 "${OUT_DIR}/smollm2_vgf_kv_fp32.pte" \
  --pte-linear8a8w "${OUT_DIR}/smollm2_vgf_kv_linear8a8w.pte" \
  --pte-linear16a8w "${OUT_DIR}/smollm2_vgf_kv_linear16a8w.pte" \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompts-file "${OUT_DIR}/wikitext_prompts_1000.txt" \
  --num-prompts 1000 \
  --ppl-prompts 100 \
  --max-seq-length 128 \
  --max-prompt-tokens 128 \
  --max-tokens-per-prompt 64 \
  --use-kv-cache \
  --refresh-prompts
```

Notes:
- This script downloads `wikitext-2-raw-v1` via Hugging Face `datasets`.
- The prompts file is reusable; omit `--refresh-prompts` on later runs.
- Perplexity is computed on the first 100 prompts from that file.
- KV-cache prompts are scored step-by-step with `token` and `input_pos`
  inputs, so this path is slower but matches the recommended decode contract.
- `--max-tokens-per-prompt` controls how many tokens are scored from each
  prompt. Use `64` for comparison with the static-window baseline, or a larger
  value to stress a longer KV context.
- For static non-KV PTEs, omit `--use-kv-cache` and pass full-logits PTEs; each
  prompt is then scored from one full-logits invocation rather than one
  invocation per token.

## 6. Notes
- The quickstart exports use KV cache. Static non-KV full-logits exports are
  still useful for faster perplexity experiments and fixed-window debugging.
- The recommended quantized export keeps KV storage in FP32. Static INT8 KV
  storage remains available for experimental memory and performance studies.
- With KV cache, generation replays one token plus `input_pos` per runner
  invocation; without KV cache, the model recomputes the full token window for
  each generated token.
- `linear8a8w` still shows noticeably more quality loss than `linear16a8w`.
- When you change `max_seq_length`, regenerate any cached prompt inputs to match the new window size.
- On hosts with multiple Vulkan devices, use `vulkaninfo --summary` to check
  device ordering and memory before selecting a non-default physical device.

### Implementation details
- The VKML runner is `examples/portable/executor_runner/executor_runner.cpp`,
  built here as `cmake-out-vkml/executor_runner`.
- `generate_sampled.py` tokenizes prompts, prepares either the fixed non-KV
  token window or KV-cache `token,input_pos` inputs, invokes `executor_runner`,
  reads logits, and decodes sampled tokens.
- The non-KV sampling and perplexity commands pass `--full-logits` to match the
  exported full-logits PTEs. KV-cache PTEs return next-token logits directly
  and should use `--use-kv-cache` instead.
