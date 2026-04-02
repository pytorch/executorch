# SmolLM2 → VGF Quickstart

> **Heads-up:** The current VGF PTQ flow is still experimental. Use FP32 as the baseline, expect `linear8a8w` to be accuracy-sensitive, and treat `linear16a8w` as the preferred quantized path to try first.

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
hf download HuggingFaceTB/SmolLM2-135M-Instruct tokenizer.json \
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

## 2. Recommended: FP32 export
Produces a stable `.pte` for experimentation and sampling.
```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_fp32 \
  export.output_name=smollm2_vgf_fp32_full_logits.pte \
  export.max_seq_length=64 \
  export.max_context_length=64 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP \
  model.use_kv_cache=False \
  model.enable_dynamic_shape=False \
  debug.verbose=True \
  debug.generate_full_logits=True
```


## 3. Experimental: 8-bit PTQ (Linear-only)
This quantizes only `torch.nn.Linear` modules using the Arm VGF PT2E quantizer.

Supported calibration inputs:
- `quantization.calibration_data=@...` for a text corpus
- `quantization.calibration_tasks=[wikitext]` for LM-Eval tasks

For this static non-KV-cache flow, keep `debug.generate_full_logits=True` for
calibrated exports. Calibration uses padded fixed-shape prefixes, and full
logits let the calibration/eval helpers select the last real-token logits row
instead of accidentally using the padded tail.

Example (LM-Eval wikitext calibration):
```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_linear8a8w \
  export.output_name=smollm2_vgf_linear8a8w_wikitext_full_logits.pte \
  export.max_seq_length=64 \
  export.max_context_length=64 \
  quantization.pt2e_quantize=vgf_8a8w \
  quantization.calibration_tasks=\[wikitext\] \
  quantization.calibration_limit=64 \
  quantization.calibration_seq_length=64 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP+INT \
  backend.vgf.quantize_scope=linear \
  model.use_kv_cache=False \
  model.enable_dynamic_shape=False \
  debug.verbose=True \
  debug.generate_full_logits=True
```

Example (16-bit activations, 8-bit weights, Linear-only):

```bash
python -m extension.llm.export.export_llm \
  base.model_class=smollm2 \
  base.params=examples/models/smollm2/135M_config.json \
  base.tokenizer_path=data/tokenizers/smollm2/tokenizer.json \
  export.output_dir=outputs/$(date +%F)/$(date +%H-%M-%S)_linear16a8w \
  export.output_name=smollm2_vgf_linear16a8w_wikitext_full_logits.pte \
  export.max_seq_length=64 \
  export.max_context_length=64 \
  quantization.pt2e_quantize=vgf_16a8w \
  quantization.calibration_tasks=\[wikitext\] \
  quantization.calibration_limit=64 \
  quantization.calibration_seq_length=64 \
  backend.vgf.enabled=True \
  backend.vgf.compile_spec=TOSA-1.0+FP+INT+int16 \
  backend.vgf.quantize_scope=linear \
  model.use_kv_cache=False \
  model.enable_dynamic_shape=False \
  debug.verbose=True \
  debug.generate_full_logits=True
```

`quantization.pt2e_quantize` selects the numeric mode.
`backend.vgf.quantize_scope=linear` keeps quantization limited to
`torch.nn.Linear` modules. The compile spec still includes FP because the rest
of the graph remains floating point.

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
`cmake-out-vkml/executor_runner`, keeps a sliding fixed-length token window,
and can print the top-5 logits each step.

Greedy generation (`--temperature 0`) always chooses the highest-logit next
token, which is useful for deterministic comparisons. Stochastic generation
(`--temperature 0.8` with `--top-p 0.9`) samples from a filtered probability
distribution, so it can produce more varied text while still being reproducible
with a fixed `--seed`.

Notes:
- `--max-seq-length` must match the export `export.max_seq_length` (otherwise you will hit input size mismatch).
- The exported SmolLM2 VGF input is `int32[1, max_seq_length]`; the helper writes
  token windows as `int32` binary inputs for `executor_runner`.
- Use `--persistent-runner` for faster multi-token generation (loads the model once).
- The documented examples use `--temperature 0` (greedy) and `--temperature 0.8`.
- For deterministic comparisons against saved `temp0` outputs, use `--seed 0`, `--repetition-penalty 1.1`, and `--no-topk-print`. At `--temperature 0`, token selection is greedy, so `--top-p` does not affect the chosen token.

Greedy example (`T=0`):
```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --persistent-runner \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_fp32_full_logits.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --max-seq-length 64 \
  --max-new-tokens 10 \
  --seed 0 \
  --temperature 0 \
  --repetition-penalty 1.1 \
  --full-logits
```

Stochastic example (`T=0.8`):
```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --persistent-runner \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_fp32_full_logits.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --max-seq-length 64 \
  --max-new-tokens 10 \
  --seed 0 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --full-logits
```
> Swap `--pte` to the quantized build to compare behaviour. `linear8a8w` still
> tends to drift more than `linear16a8w`.



### 4.2 Batch prompts from `default_prompts.txt`

To generate for *all* prompts in `default_prompts.txt` and save to a file:

```bash
python examples/arm/smollm2_example_vgf/generate_sampled.py \
  --persistent-runner \
  --runner cmake-out-vkml/executor_runner \
  --pte smollm2_vgf_fp32_full_logits.pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt-file examples/arm/smollm2_example_vgf/default_prompts.txt \
  --prompt-all \
  --max-seq-length 64 \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --full-logits \
  --save-generations outputs/$(date +%F)/$(date +%H-%M-%S)_smollm2_gen.txt
```

## 5. Wikitext prompts and perplexity

Build a reusable 1000-prompt file from `wikitext-2-raw-v1` and evaluate
perplexity on the first 100 prompts for FP32, `linear8a8w`, and `linear16a8w`:

```bash
OUT_DIR=outputs/$(date +%F)/$(date +%H-%M-%S)_smollm2_vgf_eval

python examples/arm/smollm2_example_vgf/eval_wikitext_perplexity.py \
  --runner cmake-out-vkml/executor_runner \
  --pte-fp32 "${OUT_DIR}/smollm2_vgf_fp32_full_logits.pte" \
  --pte-linear8a8w "${OUT_DIR}/smollm2_vgf_linear8a8w_wikitext_full_logits.pte" \
  --pte-linear16a8w "${OUT_DIR}/smollm2_vgf_linear16a8w_wikitext_full_logits.pte" \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompts-file "${OUT_DIR}/wikitext_prompts_1000.txt" \
  --num-prompts 1000 \
  --ppl-prompts 100 \
  --max-seq-length 64 \
  --max-prompt-tokens 64 \
  --refresh-prompts
```

Notes:
- This script downloads `wikitext-2-raw-v1` via Hugging Face `datasets`.
- The prompts file is reusable; omit `--refresh-prompts` on later runs.
- Perplexity is computed on the first 100 prompts from that file.
- Each prompt is capped to 64 tokens and scored from one full-logits
  `executor_runner` invocation per prompt, rather than one invocation per token.

## 6. Notes
- This flow keeps KV cache disabled and uses a fixed token window. KV-cache
  support is the expected next step for faster generation, but it is outside
  this static VGF quickstart.
- Without KV cache, the model recomputes the entire token window for each
  generated token.
- `linear8a8w` still shows noticeably more quality loss than `linear16a8w`.
- When you change `max_seq_length`, regenerate any cached prompt inputs to match the new window size.
- On hosts with multiple Vulkan devices, use `vulkaninfo --summary` to check
  device ordering and memory before selecting a non-default physical device.

### Implementation details
- The VKML runner is `examples/portable/executor_runner/executor_runner.cpp`,
  built here as `cmake-out-vkml/executor_runner`.
- `generate_sampled.py` tokenizes prompts, prepares the fixed token window,
  invokes `executor_runner`, reads logits, and decodes sampled tokens.
- The sampling and perplexity commands pass `--full-logits` to match the
  exported full-logits PTEs.
