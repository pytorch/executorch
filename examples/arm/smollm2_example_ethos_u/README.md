# SmolLM2 -> Ethos-U Quickstart

> **Heads-up:** This Ethos-U post-training quantization flow is still
> experimental. The current recommended path is `w8a16` with
> `quantization.quantize_scope=linear`, which places the linear layers on
> Ethos-U while the remaining FP32 operators still run on the Corstone-320 FVP
> host CPU. That hybrid setup is deliberate: it is the simplest path in this
> example that still produces meaningful text.
>
> This example exports the base `HuggingFaceTB/SmolLM2-135M` checkpoint via
> `base.model_class=smollm2`, so fetch the matching tokenizer from the same
> model family. Do not mix this flow with the `SmolLM2-135M-Instruct`
> tokenizer/checkpoint pair unless you intentionally change the exported model.

This document focuses on one validated flow:

1. Export one generation-ready full-logits `w8a16` PTE with a fixed sequence window of 32.
2. Build one runner that embeds that PTE and uses semihosting for host-side
   input/output tensor exchange.
3. Run a short prompt-generation smoke test on Corstone-320 FVP.
4. Optionally evaluate Wikitext perplexity with the same full-logits artifact.

In this example, semihosting is mainly a convenient FVP integration path for
passing meaningful input tensors into the runner and reading output tensors back
out. The Python host script does the tokenization and prompt preprocessing, then
uses semihosting to provide the resulting input tensor to the model and collect
the output logits. Embedding the PTE is a separate convenience that avoids
copying the model file at runtime. On real silicon, the same preprocessing would
more likely populate the model input buffer directly from software rather than
via semihosting.

The example uses a fixed sequence length of 32 because that is the current
validated tradeoff for this branch on Corstone-320 FVP. Larger windows were more
expensive in runtime and stalled in our experiments, while smaller windows were
easier to validate earlier but produced weaker prompts and less representative
perplexity results. This branch also does not use KV-cache decoding, so every
generated token recomputes attention across the whole window and larger sequence
lengths become even more costly. If KV-cache support is added later, it should
reduce the incremental decode cost, but it is not the direct reason seq32 was
chosen here.

## 0. Prerequisites

Run all commands from the repository root.

Use an activated Python environment before running the setup commands below,
because `examples/arm/setup.sh` installs Python packages into the active
environment. A conda environment or Python `venv` both work; see
[`docs/source/using-executorch-building-from-source.md`](../../../docs/source/using-executorch-building-from-source.md)
for the general ExecuTorch environment setup.

```bash
cd /path/to/executorch
source /path/to/venv/bin/activate
```

Install the Arm Ethos-U dependencies and generate `setup_path.sh`:

```bash
examples/arm/setup.sh \
  --i-agree-to-the-contained-eula \
  --enable-ethos-u-deps
```

Source the generated Arm setup:

```bash
source examples/arm/arm-scratch/setup_path.sh
```

Install the helper Python packages used by this example:

```bash
pip install -U "huggingface_hub[cli]" datasets
pip install -e ./extension/llm/tokenizers/
```

Build the ExecuTorch Arm libraries once so the runner wrappers can find the
`executorch` package in `arm_test`:

```bash
bash backends/arm/scripts/build_executorch.sh
```

If you want the broader Arm backend setup flow, see `examples/arm/README.md`.

## 1. Tokenizer

Download the tokenizer that matches the exported base SmolLM2 checkpoint:

```bash
mkdir -p data/tokenizers/smollm2
hf download HuggingFaceTB/SmolLM2-135M tokenizer.json \
  --local-dir data/tokenizers/smollm2
```

## 2. Recommended configuration

These are the settings used by the main flow in this README:

- `quantization.pt2e_quantize=ethosu_16a8w`
- `quantization.quantize_scope=linear`
- `export.max_seq_length=32`
- `export.max_context_length=32`
- `quantization.calibration_seq_length=32`
- `quantization.calibration_limit=62`
- `backend.ethosu.target=ethos-u85-256`
- `backend.ethosu.system_config=Ethos_U85_SYS_DRAM_High`
- `backend.ethosu.memory_mode=Dedicated_Sram_512KB`

Why these settings matter:

- `linear` scope means only the linear layers are quantized onto Ethos-U. This
  is the current validated path for meaningful output in this example.
- `max_seq_length=32` and `calibration_seq_length=32` are kept equal so the
  quantizer observes the same token-window shape that the runtime will execute.
  Keeping them aligned avoids calibrating a shape that the deployed runner never
  uses.
- `calibration_limit=62` is the current fuller-calibration setting for this
  README. With the newer full-logits calibration path, larger limits are now
  practical enough to use by default. For quicker iteration, `calibration_limit=2`
  is the fast validation setting discussed later in this document.

## 3. Export the generation artifact

This command produces the full-logits PTE used for the generation smoke test and optional perplexity evaluation. Static non-KV calibration uses padded prefixes, so calibrated exports must produce full logits to let calibration select the last real token position instead of a padded position.

```bash
bash examples/arm/smollm2_example_ethos_u/export_smollm2_ethosu.sh \
  --mode=w8a16 \
  --max_seq_length=32 \
  --max_context_length=32 \
  --calibration_limit=62 \
  --calibration_seq_length=32 \
  --quantize_scope=linear
```

What this command does:

- `--mode=w8a16` selects the 16-bit activation, 8-bit weight Ethos-U quantizer.
- By default the helper writes the exported `.pte` into the repository root, so
  the runner build commands below can reference the artifact by filename.
- `--max_seq_length=32` fixes the deployed token window to 32 tokens.
- `--max_context_length=32` keeps prompt context management consistent with that
  same fixed window.
- `--calibration_limit=62` uses the fuller calibration setting now recommended
  for this example.
- `--calibration_seq_length=32` calibrates on the same token length that the
  runtime will execute.
- `--quantize_scope=linear` keeps the validated hybrid setup where linear layers
  run on Ethos-U and the rest of the graph remains FP32.

The output artifact is named:

```text
smollm2_ethosu_seq32_w8a16_wikitext_full_logits.pte
```

## 4. Build the semihosting runner

Build one runner that embeds the generation artifact:

```bash
bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte=smollm2_ethosu_seq32_w8a16_wikitext_full_logits.pte \
  --output=smollm2_ethosu_seq32_w8a16_wikitext_full_logits/cmake-out \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00400000 \
  --input_file_pool_size=0x00100000
```

What this command does:

- Builds a semihosting `arm_executor_runner` ELF so the host can pass
  preprocessed input tensors in and read output tensors back out easily on FVP.
  In this flow the PTE is embedded in that runner as a separate convenience.
- Uses the validated `Ethos_U85_SYS_DRAM_High` and `Dedicated_Sram_512KB`
  defaults from the build helper, so you do not need to pass them explicitly in
  the common case.
- Sets three allocator pool sizes that keep the embedded-PTE full-logits runner inside a
  practical Corstone-320 DDR budget.

How to read the pool sizes:

- `method_pool_size` stores long-lived runtime objects such as the loaded
  method and model state.
- `scratch_pool_size` is temporary workspace used during execution.
- `input_file_pool_size` is the buffer used to load semihosted input files such
  as `i0.bin`.

These values are not universal tuning rules. They are simply the validated pool
sizes for this example's seq32 embedded-PTE runner. Start with them unless you
are actively changing the export shape or runtime integration.

## 5. Run a generation smoke test

Use `generate_sampled.py` to tokenize the prompt on the host, write the input
tensor file expected by the semihosting runner, launch FVP, read back the
output logits, and decode the generated token IDs into text:

```bash
python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner smollm2_ethosu_seq32_w8a16_wikitext_full_logits/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --window 32 \
  --max-new-tokens 2 \
  --full-logits \
  --temperature 0 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

How to interpret the main options:

- `--embedded-pte` tells the script not to copy a separate `program.pte`,
  because the runner already contains the model.
- `--window 32` must match the exported `max_seq_length`. If these differ, the
  runner will reject the input tensor shape.
- `--max-new-tokens 2` keeps the smoke test short. The goal here is to show the
  end-to-end path works, not to benchmark long decoding.
- `--full-logits` tells `generate_sampled.py` to select the last valid prompt
  row from the `[window, vocab]` output. This matches the calibrated static
  non-KV export path and avoids sampling from padded positions.
- `--temperature 0` switches to greedy decoding, which is the most stable way
  to compare short smoke runs.
- `--top-p 0.9` is kept for consistency with the broader sampling interface,
  but it does not affect greedy decoding when `--temperature 0`.
- `--repetition-penalty 1.1` still matters in greedy mode because it modifies
  the logits before `argmax`.

## 6. Optional: evaluate Wikitext perplexity

The calibrated generation artifact already returns full logits for every token position in the 32-token window, so the same PTE and runner can be used for perplexity scoring.

### 6.1 Build the matching runner

```bash
bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte=smollm2_ethosu_seq32_w8a16_wikitext_full_logits.pte \
  --output=smollm2_ethosu_seq32_w8a16_wikitext_full_logits/cmake-out \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00400000 \
  --input_file_pool_size=0x00100000
```

The full-logits artifact uses `--method_pool_size=0x01000000` (`16 MiB`).

### 6.2 Run perplexity

```bash
python examples/arm/smollm2_example_ethos_u/eval_wikitext_perplexity.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner-w8a8 smollm2_ethosu_seq32_w8a16_wikitext_full_logits/cmake-out/arm_executor_runner \
  --runner-w8a16 smollm2_ethosu_seq32_w8a16_wikitext_full_logits/cmake-out/arm_executor_runner \
  --prompts-file outputs/$(date +%F)/wikitext_prompts_seq32.txt \
  --num-prompts 100 \
  --ppl-prompts 100 \
  --min-prompt-tokens 32 \
  --max-prompt-tokens 32 \
  --max-tokens-per-prompt 32 \
  --window 32 \
  --timeout 36000 \
  --refresh-prompts
```

Why the prompt settings are all 32 here:

- `--window 32` must match the export shape.
- `--min-prompt-tokens 32` and `--max-prompt-tokens 32` force every prompt to
  fill exactly one scoring window, which makes the comparison easier to reason
  about.
- `--max-tokens-per-prompt 32` keeps scoring aligned with that same fixed
  window.
- `--num-prompts 100` builds a reusable prompt file with enough samples for a
  stable comparison.
- `--ppl-prompts 100` then scores all prompts from that file. Lower this value
  when you want a quicker but noisier local check.

The evaluator script compares two runners, which is why it asks for both
`--runner-w8a8` and `--runner-w8a16`. In this simplified `w8a16`-only flow, it
is acceptable to pass the same runner to both options when you only want one
number from the validated artifact.

## 7. Additional notes

### Why padding is needed for full-logits evaluation

The full-logits export returns one logits row per position in the fixed window.
Short prompts therefore need padding so the runtime still receives a tensor with
exactly 32 token slots. For perplexity, the evaluator right-pads the prompt so
the real tokens stay at the front of the causal window and each target token is
scored against the matching row. This preserves the usual left-to-right causal
ordering even though the deployed runtime works with fixed-size inputs.

### What `full` quantization scope means

`quantization.quantize_scope=full` asks the export stack to quantize more than
just the linear layers. That path exists for experimentation, but it is not the
validated path in this README because the linear-only setup is the one that
currently produces the clearest end-to-end result on Ethos-U FVP.

### Can calibration be faster?

Yes. The quickest way to iterate is to lower `--calibration_limit`. The tradeoff
is that you are collecting activation statistics from fewer samples, which can
hurt perplexity and generation quality. Keep `--calibration_seq_length` aligned
with `--max_seq_length`; if they differ, the calibration run is no longer
measuring the same tensor shapes that the deployed model will execute. In the
older non-KV path, calibration was especially slow because it often replayed
many partial prefixes position by position. The newer full-logits path can
observe a whole 32-token window in one pass, so larger limits are now much more
practical.

In the saved seq32 runs in this branch, `--calibration_limit=62` is now
bearable as the fuller-calibration setting, while `--calibration_limit=2`
remains the fast validation option. On the 100-prompt perplexity check, `2`
scored best, but `62` was still competitive and is the more conservative
default when export turnaround is less important than fuller calibration.

### Historical seq8 artifacts

Earlier experiments in this directory used smaller seq8 exports and separate
included-PTE runners. They are useful as implementation history, but they are
not the main path for this README because they add options without improving the
clarity of the validated seq32 `w8a16` workflow.

### Clean-checkout checklist

If the example fails on a clean checkout, the most common missing pieces are:

- `huggingface_hub[cli]` for the `hf download` command.
- `datasets` for rebuilding Wikitext prompts in the perplexity script.
- `pytorch_tokenizers`, installed from `./extension/llm/tokenizers/`.
- `backends/arm/scripts/build_executorch.sh`, which populates the default
  `arm_test` build root used by the runner wrappers.
