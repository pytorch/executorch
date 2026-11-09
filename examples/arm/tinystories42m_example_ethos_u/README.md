# TinyStories-42M KV-cache generation on Ethos-U85

This example exports the `stories42M.pt` checkpoint as a static-shape,
token-at-a-time KV-cache model with linear `w8a16` PT2E quantization. It reuses
the validated SmolLM2 Ethos-U semihosting runner and host generation helper.

## Set up

From the ExecuTorch repository root:

```bash
source ../optimum-executorch/.venv_et/bin/activate
source examples/arm/arm-scratch/setup_path.sh

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX="${CXX}"
```

Place these files in one directory:

```text
data/tinystories42m/stories42M.pt
data/tinystories42m/params.json
data/tinystories42m/tokenizer.model
```

The checkpoint is available from
`https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt`.
Set `TINYSTORIES42M_ARTIFACT_DIR` when the files live elsewhere.

Use these parameters for the 42M checkpoint:

```json
{
  "dim": 512,
  "multiple_of": 32,
  "n_heads": 8,
  "n_kv_heads": 8,
  "n_layers": 8,
  "norm_eps": 1e-05,
  "vocab_size": 32000
}
```

## Export ctx64 plain KV cache

```bash
bash examples/arm/tinystories42m_example_ethos_u/export_tinystories42m_ethosu.sh
```

The default export uses:

```text
model.use_kv_cache=True
model.quantize_kv_cache=False
model.enable_dynamic_shape=False
export.max_seq_length=64
export.max_context_length=64
quantization.pt2e_quantize=ethosu_16a8w
quantization.quantize_scope=linear
quantization.calibration_tasks=[wikitext]
quantization.calibration_limit=1
quantization.calibration_seq_length=64
debug.generate_full_logits=False
```

Although the KV model consumes one token per invocation, `max_seq_length=64`
describes the exported sequence capacity. PT2E uses one 64-token Wikitext
sample for the quick validation calibration.

## KV-cache configuration

Use FP32 KV-cache buffers for Alif E8 deployment. These buffers retain the key
and value tensors computed for earlier tokens and are updated for each new
token. Board measurements found decoding with FP32 KV-cache buffers
approximately 6x faster than with statically quantized int8 KV-cache buffers.
At ctx64, storing the KV-cache buffers as int8 reduces their runtime size from
approximately 2 MiB to 0.5 MiB, but does not reduce the PTE stored in OSPI. For
implementation details, see the
[SmolLM2 static KV-cache section](../smollm2_example_ethos_u/README.md#7-experimental-static-quantized-kv-cache).

## Alif E8 deployment

The exported PTE is approximately 103 MiB, so it fits within the current
125 MiB model-storage budget for the Alif E8 Ethos-U85 configuration. Deploy
the PTE with the Alif board-specific ExecuTorch firmware runner; the embedded
Corstone-320 semihosting runner below is only for FVP validation and is not the
Alif firmware image.

The board integration must provide the token and `input_pos` int32 inputs,
retain the KV-cache state between decoding steps, and perform token selection
and decoding in the application. Firmware, runtime arenas, KV-cache buffers,
and tokenizer storage are additional to the PTE size and must be checked
against the board's runtime memory configuration.

## Build the FVP runner

```bash
ARTIFACT_DIR="${TINYSTORIES42M_ARTIFACT_DIR:-data/tinystories42m}"

bash examples/arm/smollm2_example_ethos_u/build_executor_runner_semihosting.sh \
  --pte="${ARTIFACT_DIR}/tinystories42m_ethosu_u85_256_ctx64_kv_w8a16.pte" \
  --output=tinystories42m_ethosu_ctx64/cmake-out \
  --target=ethos-u85-256 \
  --system_config=Ethos_U85_SYS_DRAM_Mid \
  --memory_mode=Sram_Only \
  --method_pool_size=0x01000000 \
  --scratch_pool_size=0x00800000 \
  --input_file_pool_size=0x00100000
```

## Greedy smoke test

```bash
ARTIFACT_DIR="${TINYSTORIES42M_ARTIFACT_DIR:-data/tinystories42m}"

python examples/arm/smollm2_example_ethos_u/generate_sampled.py \
  --fvp=examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  --runner=tinystories42m_ethosu_ctx64/cmake-out/arm_executor_runner \
  --embedded-pte \
  --tokenizer="${ARTIFACT_DIR}/tokenizer.model" \
  --prompt="Once upon a time in a small village," \
  --window=64 \
  --max-context-length=64 \
  --use-kv-cache \
  --max-new-tokens=2 \
  --temperature=0 \
  --top-p=0.9 \
  --repetition-penalty=1.1 \
  --timeout=1200
```

This smoke test validates export, runner integration, stateful token/position
inputs, and decoding. Add prompt-set generation and perplexity only after this
path produces a coherent continuation.
