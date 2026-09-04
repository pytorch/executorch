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

1. Export one generation-ready KV-cache `w8a16` PTE with a 64-token context.
2. Build one runner that embeds that PTE and uses semihosting for host-side
   input/output tensor exchange.
3. Run a short prompt-generation smoke test on Corstone-320 FVP.
4. Optionally evaluate Wikitext perplexity with the same KV-cache artifact.

In this example, semihosting is mainly a convenient FVP integration path for
passing meaningful input tensors into the runner and reading output tensors back
out. The Python host script does the tokenization and prompt preprocessing, then
uses semihosting to provide the resulting input tensor to the model and collect
the output logits. Embedding the PTE is a separate convenience that avoids
copying the model file at runtime. On real silicon, the same preprocessing would
more likely populate the model input buffer directly from software rather than
via semihosting.

The primary path below uses KV-cache decoding with a 64-token context. Each
runtime step executes one token plus its `input_pos`, while the exported context
length controls the KV-cache capacity. The older static non-KV seq32 full-logits
flow remains the safest fallback when debugging full-window execution. Larger
non-KV windows are experimental: they are more expensive at runtime and require
retuning the runner allocator pools. For example, a seq48 embedded-PTE smoke run
needed a scratch pool just under 5 MiB instead of the 4 MiB pool used by the
seq32 non-KV commands.

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
- `model.use_kv_cache=True`
- `export.max_seq_length=64`
- `export.max_context_length=64`
- `quantization.calibration_seq_length=64`
- `quantization.calibration_limit=1`
- `backend.ethosu.target=ethos-u85-256`
- `backend.ethosu.system_config=Ethos_U85_SYS_DRAM_High`
- `backend.ethosu.memory_mode=Dedicated_Sram_512KB`

Why these settings matter:

- `linear` scope means only the linear layers are quantized onto Ethos-U. This
  is the current validated path for meaningful output in this example.
- `model.use_kv_cache=True` exports the token and `input_pos` inputs used for
  step-wise decoding.
- `max_seq_length=64` and `max_context_length=64` set the documented 64-token
  generation context. In KV-cache mode, `generate_sampled.py` can derive the
  context length from `--window 64`; pass `--max-context-length 64` explicitly
  when you want the command line to make that relationship obvious.
- `calibration_limit=1` keeps this experimental KV-cache flow practical for
  local iteration. Increase it when you want more calibration coverage.

## 3. Export the generation artifact

This command produces the KV-cache PTE used for the generation smoke test and optional perplexity evaluation.

```bash
bash examples/arm/smollm2_example_ethos_u/export_smollm2_ethosu.sh \
  --mode=w8a16 \
  --use-kv-cache \
  --max_seq_length=64 \
  --max_context_length=64 \
  --calibration_limit=1 \
  --calibration_seq_length=64 \
  --quantize_scope=linear
```

What this command does:

- `--mode=w8a16` selects the 16-bit activation, 8-bit weight Ethos-U quantizer.
- By default the helper writes the exported `.pte` into the repository root, so
  the runner build commands below can reference the artifact by filename.
- `--use-kv-cache` exports token and `input_pos` inputs and disables full-logits
  output naming.
- `--max_seq_length=64` and `--max_context_length=64` set the documented
  64-token KV-cache context.
- `--calibration_limit=1` is the quick validation setting used for this
  experimental KV-cache path.
- `--calibration_seq_length=64` calibrates with the same context length used at
  runtime.
- `--quantize_scope=linear` keeps the validated hybrid setup where linear layers
  run on Ethos-U and the rest of the graph remains FP32.

The output artifact is named:

```text
smollm2_ethosu_kv_seq64_w8a16_wikitext.pte
```

## 4. Build the semihosting runner

Build one runner that embeds the generation artifact:

```bash
bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte=smollm2_ethosu_kv_seq64_w8a16_wikitext.pte \
  --output=smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00800000 \
  --input_file_pool_size=0x00100000
```

What this command does:

- Builds a semihosting `arm_executor_runner` ELF so the host can pass
  preprocessed input tensors in and read output tensors back out easily on FVP.
  In this flow the PTE is embedded in that runner as a separate convenience.
- Uses the validated `Ethos_U85_SYS_DRAM_High` and `Dedicated_Sram_512KB`
  defaults from the build helper, so you do not need to pass them explicitly in
  the common case.
- Sets three allocator pool sizes that keep the embedded-PTE KV-cache runner
  inside a practical Corstone-320 DDR budget.

How to read the pool sizes:

- `method_pool_size` stores long-lived runtime objects such as the loaded
  method and model state.
- `scratch_pool_size` is temporary workspace used during execution.
- `input_file_pool_size` is the buffer used to load semihosted input files such
  as `i0.bin`.

These values are not universal tuning rules. They are simply the validated pool
sizes for this example's seq64 KV-cache embedded-PTE runner. Start with them
unless you are actively changing the export shape or runtime integration.

## 5. Run a generation smoke test

Use `generate_sampled.py` to tokenize the prompt on the host, write the input
tensor file expected by the semihosting runner, launch FVP, read back the
output logits, and decode the generated token IDs into text:

```bash
python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --max-new-tokens 2 \
  --temperature 0 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

How to interpret the main options:

- `--embedded-pte` tells the script not to copy a separate `program.pte`,
  because the runner already contains the model.
- `--window 64` provides the main context length used by the script.
- `--max-context-length 64` makes the KV-cache capacity explicit; omitting it
  would use `--window`.
- `--use-kv-cache` runs the two-input token/position path and keeps one FVP
  process alive in semihosting server mode for step-wise decoding.
- `--max-new-tokens 2` keeps the smoke test short. The goal here is to show the
  end-to-end path works, not to benchmark long decoding.
- `--temperature 0` switches to greedy decoding, which is the most stable way
  to compare short smoke runs.
- `--top-p 0.9` is kept for consistency with the broader sampling interface,
  but it does not affect greedy decoding when `--temperature 0`.
- `--repetition-penalty 1.1` still matters in greedy mode because it modifies
  the logits before `argmax`.

### 5.1 Profile prompt processing and decoding

The runner exposes Ethos-U85 PMU counters for every server-mode inference.
Capture those counters with FVP fast mode disabled:

```bash
python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner smollm2_ethosu_static_kvq_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --max-new-tokens 2 \
  --temperature 0 \
  --no-topk-print \
  --profile-output outputs/ethosu_u85_profile.json \
  --timeout 24000
```

`--profile-output` accepts `.json` or `.csv` and automatically removes the
Ethos-U `--fast` FVP option. The report contains one NPU PMU sample per model
execution, split into `prefill` and `decode` phases. Passing
`--npu-frequency-mhz` also reports an estimated NPU-only token rate.

This KV-cache demo processes the prompt one token at a time; its prefill result
is therefore the aggregate and average of those token executions, not a
batched-prefill measurement. The final prompt execution supplies the logits for
the first generated token, so steady-state decode executions start with the
second generated token.

CS-320 provides useful Ethos-U85 NPU cycle and event estimates when its timing
adapters match the target configuration. Cortex-M85 CPU timing and simulator
wall time are not cycle accurate, so the report must not be interpreted as
end-to-end latency or measured device tokens/s. Use FPGA or hardware for those
measurements.

## 6. Optional: evaluate Wikitext perplexity

The KV-cache generation artifact can also be used for step-wise perplexity scoring over the same 64-token context.

### 6.1 Build the matching runner

```bash
bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte=smollm2_ethosu_kv_seq64_w8a16_wikitext.pte \
  --output=smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00800000 \
  --input_file_pool_size=0x00100000
```

The KV-cache artifact uses `--method_pool_size=0x01000000` (`16 MiB`).

### 6.2 Run perplexity

```bash
python examples/arm/smollm2_example_ethos_u/eval_wikitext_perplexity.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner-w8a8 smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --runner-w8a16 smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --prompts-file outputs/$(date +%F)/wikitext_prompts_kv_seq64.txt \
  --num-prompts 100 \
  --ppl-prompts 50 \
  --min-prompt-tokens 64 \
  --max-prompt-tokens 64 \
  --max-tokens-per-prompt 64 \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --timeout 2400 \
  --refresh-prompts
```

Why the prompt settings are all 64 here:

- `--window 64` and `--max-context-length 64` match the exported KV-cache
  context.
- `--min-prompt-tokens 64` and `--max-prompt-tokens 64` force every prompt to
  fill one scoring context, which makes the comparison easier to reason about.
- `--max-tokens-per-prompt 64` keeps scoring aligned with that same context.
- `--num-prompts 100` builds a reusable prompt file with enough samples for a
  stable comparison.
- `--ppl-prompts 50` keeps the FVP check manageable. Raise it when you want a
  fuller but slower run.

The evaluator script compares two runners, which is why it asks for both
`--runner-w8a8` and `--runner-w8a16`. In this simplified `w8a16`-only flow, it
is acceptable to pass the same runner to both options when you only want one
number from the validated artifact. A 50-prompt seq64 KV-cache run in this
branch completed with perplexity around 45.

## 7. Experimental static quantized KV cache

Ethos-U does not support the dynamic per-token quantization used by
`model.quantize_kv_cache=True`. The static path instead calibrates fixed
per-channel K/V scales from Wikitext before export, stores the cache as int8,
and uses standard tensor cache updates. It does not link the Llama custom cache
operator, and runner inputs remain int32.

Export the seq64 `w8a16` static-KVQ artifact with one Wikitext calibration
sample:

```bash
bash examples/arm/smollm2_example_ethos_u/export_smollm2_ethosu.sh \
  --mode=w8a16 \
  --use-kv-cache \
  --static-quantize-kv-cache \
  --so_library=/path/to/cmake-out/kernels/quantized/libquantized_ops_aot_lib.so \
  --max_seq_length=64 \
  --max_context_length=64 \
  --calibration_limit=1 \
  --calibration_seq_length=64 \
  --quantize_scope=linear
```

`--so_library` is needed only during export to register portable QDQ out
variants. Build it with `EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON` if your
ExecuTorch installation does not already provide it. The embedded runner uses
the standard `quantized_ops_lib`; it does not link Llama custom cache operators.

This produces:

```text
smollm2_ethosu_static_kvq_seq64_w8a16_wikitext.pte
```

Build it with the standard semihosting runner; no custom cache-op source or
linkage is required:

```bash
bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte=smollm2_ethosu_static_kvq_seq64_w8a16_wikitext.pte \
  --output=smollm2_ethosu_static_kvq_seq64_w8a16_wikitext/cmake-out \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00800000 \
  --input_file_pool_size=0x00100000
```

Run the ten-token greedy smoke test:

```bash
python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner smollm2_ethosu_static_kvq_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt "Once upon a time in a small village," \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --max-new-tokens 10 \
  --temperature 0 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

Run all nine default prompts greedily and save the generated text:

```bash
python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner smollm2_ethosu_static_kvq_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer data/tokenizers/smollm2/tokenizer.json \
  --prompt-file examples/arm/smollm2_example_ethos_u/default_prompts.txt \
  --prompt-all \
  --save-generations outputs/$(date +%F)/ethosu_static_kvq_seq64_generations.txt \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --max-new-tokens 64 \
  --temperature 0 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --timeout 2400
```

For a 50-prompt comparison, pass the plain-KV runner as `w8a8` and the
static-KVQ runner as `w8a16`; those labels are only result keys in this helper:

```bash
python examples/arm/smollm2_example_ethos_u/eval_wikitext_perplexity.py \
  --runner-w8a8 smollm2_ethosu_kv_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --runner-w8a16 smollm2_ethosu_static_kvq_seq64_w8a16_wikitext/cmake-out/arm_executor_runner \
  --prompts-file outputs/$(date +%F)/wikitext_prompts_kv_seq64.txt \
  --num-prompts 100 \
  --ppl-prompts 50 \
  --min-prompt-tokens 64 \
  --max-prompt-tokens 64 \
  --max-tokens-per-prompt 64 \
  --window 64 \
  --max-context-length 64 \
  --use-kv-cache \
  --timeout 24000
```

## 8. Additional notes

### Non-KV fallback: why padding is needed for full-logits evaluation

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

### Non-KV fallback: can calibration be faster?

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

### Clean-checkout checklist

If the example fails on a clean checkout, the most common missing pieces are:

- `huggingface_hub[cli]` for the `hf download` command.
- `datasets` for rebuilding Wikitext prompts in the perplexity script.
- `pytorch_tokenizers`, installed from `./extension/llm/tokenizers/`.
- `backends/arm/scripts/build_executorch.sh`, which populates the default
  `arm_test` build root used by the runner wrappers.
