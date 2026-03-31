# Qwen3-TTS Single-PTE Unified Export — Progress

## Goal
Replace the multi-bucket decoder-only pipeline with a single `.pte` file containing
all pipeline stages (text→audio), deployable on iOS/Android following the Parakeet pattern.

## Architecture

Single `model.pte` with 6 named methods + constant metadata:

| Method | Input | Output | Shapes |
|---|---|---|---|
| `encode_text` | `token_ids [1, S]` | `projected [1, S, 1024]` | Dynamic S |
| `talker` | `embeds [1, S, 1024], input_pos [S]` | `logits [1, 3072], hidden [1, 1024]` | Dynamic S |
| `code_predictor` | `embeds [1, S, 1024], input_pos [S]` | `hidden [1, 1024]` | Dynamic S |
| `codec_embed` | `token_id [1], group_idx [1]` | `embed [1, 1, 1024]` | Static |
| `cp_head` | `hidden [1, 1024], head_idx [1]` | `logits [1, 2048]` | Static |
| `decode_audio` | `codes [1, T, 16]` | `wav [1, T*1920], lengths [1]` | Dynamic T |

Constant methods: `output_sample_rate=24000`, `num_quantizers=16`, `max_seq_len=256`,
`talker_vocab_size=3072`, `num_code_groups=16`, `talker_dim=1024`.

## Runner Orchestration (C++)

```
text → tokenize (tiktoken C++) → encode_text → projected text embeds
→ assemble composite prefill: (codec control tokens + speaker + text) embeddings summed
→ talker(prefill) → logits_0, hidden_0
→ loop until codec_eos:
    sample code_0 from logits
    codec_embed(code_0, group=0) → main embed
    code_predictor(prefill=[hidden, main_embed], pos=[0,1])
    for i in 1..15:
        cp_head(cp_hidden, head_idx=i-1) → sample code_i
        codec_embed(code_i, group=i) → cp embed
        code_predictor(step=cp_embed, pos=[i+1])
    sum all 16 embeddings + next text embed → next_input
    talker(decode_step=next_input) → logits, hidden
→ decode_audio(accumulated_codes) → waveform → WAV file
```

## Progress

### Step 1: Understand Architecture ✅
- Studied Parakeet multi-method export pattern (export_parakeet_tdt.py)
- Analyzed Qwen3-TTS generate loop (composite embedding, code predictor, streaming text)
- Mapped all aux weights: text_embedding [151936,2048], text_projection MLP,
  main_codec_embedding [3072,1024], codec_head [3072,1024],
  15× cp_codec_embedding [2048,1024], 15× cp_lm_head [2048,1024]

### Step 2: Unified Export Script ✅
- Created `export_unified.py` with 6 wrapper modules
- Key fixes:
  - `DynamicDecoderExport`: patches CausalConvNet `math.ceil` → integer ceiling division
    for torch.export SymInt compatibility
  - `TalkerExport`: uses `apply_output=False` + separate `codec_head` Linear to return
    both logits AND hidden states
  - `CodecEmbedExport`: stacks main + 15 cp embeddings into [16, 3072, 1024] with padding,
    uses `torch.index_select` for group-based lookup
  - `CpHeadExport`: stacks 15 per-group LM heads into [15, 2048, 1024],
    uses `torch.index_select` for head selection
- Trace fix: sample inputs must use seq_len > 1 (used 4) to avoid torch.export
  specializing dynamic dims to constants

### Step 2a: Portable FP32 Export Test ✅
- Export command:
  ```
  python export_unified.py --backend portable --dtype fp32
  ```
- Result: `model_test.pte` = 3,951.8 MB (expected — fp32 unquantized)
- All 6 methods verified working:
  - `encode_text`: [1,5] → [1,5,1024] ✓
  - `talker` prefill: [1,5,1024] → logits [1,3072] + hidden [1,1024] ✓
  - `talker` decode: [1,1,1024] → logits [1,3072] + hidden [1,1024] ✓
  - `codec_embed`: token_id=42, group=0 → [1,1,1024] ✓
  - `cp_head`: hidden [1,1024], head=0 → logits [1,2048] ✓
  - `code_predictor`: [1,2,1024] → hidden [1,1024] ✓
  - `decode_audio`: [1,10,16] → wav [1,19200] + lengths ✓
  - All constant methods return correct values ✓

### Step 2b: XNNPACK 8da4w Quantized Export ✅
- Export command:
  ```
  python export_unified.py \
    --converted-dir qwen3_tts_artifacts \
    --talker-dir qwen3_tts_artifacts/talker_converted \
    --output-dir qwen3_tts_exports_unified \
    --backend xnnpack --dtype fp32 --qlinear 8da4w
  ```
- Result: `model.pte` = **2,065.4 MB** (single file, all 6 methods)
- All methods verified on quantized model:
  - `encode_text`: [1,5] → [1,5,1024] ✓
  - `talker` prefill: [1,5,1024] → logits [1,3072] + hidden [1,1024] ✓
  - `talker` decode: [1,1,1024] → logits [1,3072] + hidden [1,1024] ✓
  - `codec_embed`: group 0 (main) and group 5 (cp) both work ✓
  - `cp_head`: head_idx=0 → logits [1,2048] ✓
  - `code_predictor` prefill: [1,2,1024] → hidden [1,1024] ✓
  - `code_predictor` step: [1,1,1024] → hidden [1,1024] ✓
  - `decode_audio`: [1,10,16] → wav [1,19200], lengths=19200 ✓
  - All constant methods verified ✓
- Size breakdown (estimated):
  - text_embedding [151936, 2048] in fp32: ~1,244 MB (NOT quantized — it's nn.Embedding)
  - talker 28L 8da4w: ~260 MB
  - code_predictor 5L 8da4w: ~52 MB
  - decoder 8da4w: ~285 MB
  - codec_embed [16, 3072, 1024] fp32: ~192 MB
  - cp_head [15, 2048, 1024] fp32: ~120 MB (buffer, not quantized)
  - KV cache buffers: ~65 MB
- **Key optimization opportunity**: text_embedding dominates at ~1.2 GB.
  Quantizing it to 8-bit would halve it to ~620 MB, bringing total to ~1.4 GB.
  Quantizing to 4-bit: ~310 MB, total ~850 MB.

### Step 2c: Embedding Quantization ✅
- Added `--qembedding` flag to `export_unified.py` (supports `4w` and `8w`)
- Embedding quantization applied only to `encode_text` module (nn.Embedding layers)
- Results:
  | Config | Size | Python test | C++ test |
  |---|---|---|---|
  | 8da4w (no emb quant) | 2,065 MB | ✅ | ✅ |
  | 8da4w + 8w embedding | 1,176 MB | ❌ (missing kernel) | ✅ (quantized_ops_lib) |
  | 8da4w + 4w embedding | 1,027 MB | ❌ (missing kernel) | ✅ (quantized_ops_lib) |
- Python pybindings lack `quantized_decomposed::embedding_byte.out` kernel,
  but C++ runner links `quantized_ops_lib` which has it
- text_embedding dropped from ~1,244 MB → ~620 MB (8w) or ~310 MB (4w)

### Comparison: Old vs New Architecture
| | Old (multi-bucket decoder) | New (unified single-PTE) |
|---|---|---|
| Files | 5× decoder .pte + talker .pte + cp .pte + aux.pth | 1× model.pte |
| Total size | ~1.4 GB (decoder only, no talker) | 1.0-2.1 GB (full pipeline) |
| Pipeline | Python talker → C++ decoder | C++ text→audio (planned) |
| Mobile ready | No (requires Python for talker) | Yes (single .pte + tokenizer) |
| Decoder speed | 3.1s (bucketed) | **2.4s** (dynamic, with warmup) |

### Step 3: C++ Unified Runner ✅
- Created `qwen3_tts_unified_runner.h/cpp` — multi-method runner
- Created `main_unified.cpp` — CLI with decode-only and text-to-audio modes
- Updated `CMakeLists.txt` — new `qwen3_tts_unified_runner` target
- Runner loads single .pte, reads metadata from constant_methods,
  loads all 6 methods by name
- Backward compat: `--codes_path` for precomputed codes decode
- Forward path: `--text` for full synthesis (tokenizer integration pending)
- **Test result** (decode-only with precomputed codes):
  ```
  Model loaded in 2607 ms
  Decoded 174720 samples (7.28s audio) in 8206 ms (0.89x realtime)
  Output: /tmp/unified_decode_test.wav (349 KB, 24kHz mono)
  ```

### Step 4: CMake Updates ✅
- Added `qwen3_tts_unified_runner` build target
- Reuses existing link libraries (XNNPACK, quantized_ops_lib, etc.)
- No new dependencies needed for decode-only mode

### Step 5: Performance Investigation & Fix ✅
- **Root cause**: XNNPACK delegate initialization on first call takes ~5.5s for the
  2 GB multi-method .pte. This penalty is paid once per method, but the C++ runner
  only calls `decode_audio` once — so it always hit this cold-start penalty.
- **Fix**: Added `warmup_decode()` that runs a 1-code dummy inference during model
  loading, triggering XNNPACK delegate init before the timed decode.
- **Results**:
  | Runner | Time | Realtime factor |
  |---|---|---|
  | Old bucketed (static 150, padding) | 3.9s | ~1.9x RT |
  | Unified (no warmup) | 8.6s | 0.84x RT |
  | Unified (with warmup) | **2.4s** | **3.05x RT** |
  | Python pybindings (reference) | 2.2s | 3.3x RT |
- Dynamic shapes process FEWER elements (91 vs 150), resulting in genuinely faster
  decode once XNNPACK is initialized
- Model load time (including warmup): 6.4s. Acceptable for app startup.

### Step 6: Remaining Work
- **Tokenizer integration**: Add tiktoken C++ tokenizer loading for `--text` mode
- **Full synthesis loop**: Implement composite prefill + autoregressive decode
  in `synthesize()` method
- **C API**: `qwen3_tts_c_api.h/cpp` for iOS/Android FFI (following Parakeet pattern)
- **Performance**: Current decode is 8.2s for 91 codes (0.89x realtime).
  The old bucketed decoder was 3.1s. Investigate why dynamic shapes are slower.
- **Model size**: Ship with 4w embedding quantization for ~1 GB total

## Parameter Counts (from export)
| Module | Parameters | Buffers |
|---|---|---|
| encode_text | 317,459,456 | 0 |
| talker | 443,613,184 | 16,646,144 |
| code_predictor | 78,655,744 | 349,184 |
| codec_embed | 0 | 50,331,648 |
| cp_head | 0 | 31,457,280 |
| decode_audio | 114,323,137 | 32 |

Note: `encode_text` is large due to text_embedding [151936, 2048] = 312M params.
With 8da4w, the text_embedding's Linear layers get quantized but the Embedding table
stays full-precision. Embedding quantization (8w) would reduce this further.


### Step 7: Architecture v2 — Fused Code Predictor 🔄 IN PROGRESS
Based on mlx-audio analysis, the v1 `synthesize()` makes 33 method calls per step
(1 talker + 15 CP + 16 embed + 1 head), each with ~2ms sync overhead = 66ms overhead/step.
mlx-audio uses lazy eval to batch everything into 1 GPU dispatch.

**Fix:** `CpGenerateExport` — unrolls the 15-step code predictor loop into a single
torch.export graph (7121 nodes). Argmax is baked in to drive the autoregressive chain.
Returns raw logits for optional C++ re-sampling.

New per-step architecture:
- `talker_step` (1 call) → logits + hidden
- C++ samples code_0
- `codec_embed` (1 call) → code_0 embedding
- `cp_generate` (1 call) → 15 sub-code logits + embedding sum
- **Total: 3 calls/step** (down from 33)
