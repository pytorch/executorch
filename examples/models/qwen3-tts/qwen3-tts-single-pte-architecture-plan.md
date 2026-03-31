# Qwen3-TTS Single-PTE Architecture Plan
## Context
The current Qwen3-TTS implementation splits the pipeline across Python (talker code generation) and C++ (decoder-only runner), with multi-bucket decoder exports totaling ~1.4 GB on disk. This is unusable for mobile deployment. The goal is a single .pte file containing all pipeline stages (talker + code predictor + decoder), with a C++ runner that takes text in and produces audio out — deployable on iOS and Android, following the proven Parakeet/Voxtral patterns already shipping in production.

## Architecture Overview
```
text (string)
  → tokenizer (C++, tiktoken)
  → token_embedding (method)
  → text_projection (method)
  → talker_prefill (method, processes full prompt)
  → talker_decode_step (method, autoregressive loop ~91 steps)
      → code_predictor_step (method, 15 sub-code predictions per step)
  → decode_audio (method, codes → waveform)
  → WAV output
```
Single model.pte with 6 named methods + constant metadata. Single tokenizer.json for tiktoken. No other files needed.

## Methods in the .pte
Method	Input	Output	Shape	Notes
token_embedding	token_ids [1, S]	embeds [1, S, 1024]	Dynamic S	Text token embedding table
text_projection	text_embeds [1, S, D_text]	projected [1, S, 1024]	Dynamic S	2-layer MLP projecting text hidden → talker hidden
talker_prefill	embeds [1, S, 1024], cache_pos [S]	logits [1, 3072], hidden [1, 1, 1024]	Dynamic S	Full-sequence KV cache fill
talker_decode_step	token [1, 1], cache_pos [1]	logits [1, 3072], hidden [1, 1, 1024]	Static	Single autoregressive step
code_predictor_step	hidden [1, 1, 1024], group_idx [1], cache_pos [1]	logits [1, 2048]	Static	Per-group sub-code prediction with baked-in embeddings/heads
decode_audio	codes [1, T, 16]	wav [1, 1, T*1920], lengths [1]	Dynamic T	Full decoder: VQ → transformer → vocoder

### constant_methods (metadata)
```
{
    "output_sample_rate": 24000,
    "num_quantizers": 16,
    "codebook_size": 2048,
    "talker_vocab_size": 3072,
    "max_seq_len": 256,
    "num_code_groups": 16,
    "dim": 1024,
}
```

## Workstreams
### 1. Fix Decoder Dynamic Shapes
Files to modify:

examples/models/qwen3-tts/model.py — new DynamicShapeDecoderExport wrapper

- Problem: CausalConvNet._get_extra_padding_for_conv1d uses math.ceil() which fails with SymInt. But all CausalConvNet instances in this decoder use stride=1, making n_frames = length (always integer) and extra_padding = 0 always.

- Fix: Override _get_extra_padding_for_conv1d on all CausalConvNet modules after loading the decoder checkpoint. For stride=1: replace with a version that computes padding algebraically without math.ceil:

def _get_extra_padding_for_conv1d_exportable(self, hidden_state):
    length = hidden_state.shape[-1]
    # For stride=1: n_frames = length, so extra_padding = 0
    # For stride>1: use torch.div with rounding_mode="ceil"
    n_frames_ceil = (length - self.kernel_size + self.padding + self.stride) // self.stride
    ideal_length = (n_frames_ceil - 1) * self.stride + (self.kernel_size - self.padding)
    return ideal_length - length

This replaces math.ceil(float_division) with integer ceiling division (a + b - 1) // b, which torch.export can trace through symbolic shapes.

Verification: Export with dynamic_shapes={"audio_codes": {1: Dim("codes_len", min=1, max=2000)}} and run with varying input lengths.

2. Unified Export Script
Files to create:

examples/models/qwen3-tts/export_unified.py — single-PTE multi-method export
Pattern: Follow Parakeet's export_all() exactly:

programs = {}
programs["token_embedding"] = export(TokenEmbeddingExport(model), ...)
programs["text_projection"] = export(TextProjectionExport(model), ...)
programs["talker_prefill"] = export(TalkerPrefillExport(model), ...)
programs["talker_decode_step"] = export(TalkerDecodeStepExport(model), ...)
programs["code_predictor_step"] = export(CodePredictorStepExport(model), ...)
programs["decode_audio"] = export(DynamicShapeDecoderExport(decoder), ...)

et = to_edge_transform_and_lower(programs, partitioner=per_method_partitioners, constant_methods=metadata)

Export wrapper modules to create:

TokenEmbeddingExport — wraps model.text_embedding (the text token embedding table). Input: token_ids [1, S], Output: embeds [1, S, text_hidden]. Dynamic S.

TextProjectionExport — wraps the 2-layer MLP text_projection (Linear → Linear with bias). Input: text_embeds [1, S, text_hidden], Output: projected [1, S, 1024]. Dynamic S.

TalkerPrefillExport — wraps the main talker transformer in prefill mode with KV cache. Input: composite embeddings [1, S, 1024] + cache_position [S]. Output: logits [1, 3072] + last hidden [1, 1, 1024]. Dynamic S (up to max_seq_len). Shares KV cache buffers with decode step.

TalkerDecodeStepExport — wraps the same talker in single-token decode mode. Input: token_id [1, 1] + cache_position [1]. Output: logits [1, 3072] + hidden [1, 1, 1024]. Static shapes. Reuses KV cache from prefill.

CodePredictorStepExport — wraps the code predictor with all 15 per-group embeddings and 15 per-group LM heads baked in. Input: hidden [1, 1, 1024] + group_idx [1] (integer 0-14) + cache_position [1]. Output: logits [1, 2048]. The forward selects the appropriate embedding/head using torch.index_select on stacked weight tensors. Has its own KV cache (5 layers, reset between main talker steps).

DynamicShapeDecoderExport — wraps the patched decoder with dynamic codes_len. Input: codes [1, T, 16]. Output: wav, lengths. Dynamic T.

Quantization strategy (per-component, before export):

token_embedding: qembedding="8w" (large vocab table, benefits from compression)
text_projection: no quantization (small 2-layer MLP)
talker_prefill/decode_step: qlinear="8da4w" (28-layer transformer, largest component)
code_predictor_step: qlinear="8da4w" (5-layer transformer)
decode_audio: qlinear="8da4w" (conv-heavy decoder)
Partitioners (per-method, XNNPACK backend):

partitioner = {
    "token_embedding": [],  # portable (embedding lookup, no benefit from XNNPACK)
    "text_projection": [XnnpackDQ(), XnnpackPartitioner()],
    "talker_prefill": [XnnpackDQ(), XnnpackPartitioner()],
    "talker_decode_step": [XnnpackDQ(), XnnpackPartitioner()],
    "code_predictor_step": [XnnpackDQ(), XnnpackPartitioner()],
    "decode_audio": [XnnpackDQ(), XnnpackPartitioner()],
}

Estimated .pte size: ~600-700 MB (talker 28L ~260 MB + code predictor 5L ~52 MB + decoder ~285 MB + aux weights ~10 MB, all 8da4w).

3. C++ Runner — Full Pipeline
Files to create/modify:

examples/models/qwen3-tts/qwen3_tts_runner.h — redesign for multi-method single .pte
examples/models/qwen3-tts/qwen3_tts_runner.cpp — full text→audio pipeline
examples/models/qwen3-tts/qwen3_tts_c_api.h — C API for iOS/Android (following Parakeet)
examples/models/qwen3-tts/qwen3_tts_c_api.cpp — C API implementation
examples/models/qwen3-tts/main.cpp — updated CLI
Runner class redesign:

class Qwen3TTSRunner {
public:
    Qwen3TTSRunner(const std::string& model_path, const std::string& tokenizer_path);

    bool synthesize(const std::string& text, const std::string& language,
                    std::vector<float>* waveform);

    // Decode-only mode (backward compat with precomputed codes)
    bool decode_codes_file(const std::string& codes_path, std::vector<float>* waveform);

    bool write_wav_file(const std::string& path, const std::vector<float>& waveform);

    int output_sample_rate() const;

private:
    std::unique_ptr<executorch::extension::Module> module_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    // Read from constant_methods
    int max_seq_len_;
    int talker_vocab_size_;
    int num_code_groups_;
    int output_sample_rate_;

    // Pipeline stages
    bool run_token_embedding(const std::vector<int64_t>& token_ids, ...);
    bool run_text_projection(/* embeddings in/out */);
    bool run_talker_prefill(/* composite embeddings, returns logits + hidden */);
    bool run_talker_decode_step(int64_t token, int32_t pos, /* returns logits + hidden */);
    bool run_code_predictor(/* hidden, returns 15 sub-codes */);
    bool run_decode_audio(const std::vector<int64_t>& codes, int32_t codes_len,
                          int32_t num_quantizers, std::vector<float>* waveform);
};

synthesize() orchestration (following Parakeet's decode loop pattern):

Tokenize text using tiktoken tokenizer in C++.
Build prompt token sequence: [role_tokens, language_tag, text_tokens, codec_bos].
Run token_embedding on text tokens.
Run text_projection to project into talker space.
Assemble composite embedding (codec BOS embedding + projected text embeddings + pad).
Run talker_prefill — fills KV cache, returns first logits + hidden.
Autoregressive loop: a. Sample main codec token from logits (greedy or top-k/top-p). b. Check for codec EOS → break. c. Run code_predictor_step 15 times (resetting its KV cache each iteration of the outer loop), collecting sub-codes. d. Store full 16-code group. e. Compute next input: sum of all 16 codec group embeddings. f. Run talker_decode_step → next logits + hidden.
Run decode_audio on accumulated codes → waveform.
Write WAV.
C API (following Parakeet's pqt_runner_create/transcribe pattern):

typedef void (*q3tts_audio_callback_t)(const float* samples, int64_t num_samples, void* user_data);

q3tts_status_t q3tts_runner_create(const q3tts_runner_config_t* config, q3tts_runner_t** out);
void q3tts_runner_destroy(q3tts_runner_t* runner);
q3tts_status_t q3tts_runner_synthesize(
    q3tts_runner_t* runner,
    const char* text,
    const char* language,
    q3tts_audio_callback_t callback,
    void* user_data);

Thread-safe via mutex, suitable for Swift/Kotlin FFI wrappers.

4. Tokenizer Integration
File: examples/models/qwen3-tts/export_unified.py — extract tokenizer during export

Qwen3-TTS uses tiktoken (same as Qwen3 LLM). During export:

Load the HF model's tokenizer.
Save as tokenizer.json alongside the .pte.
C++ runner loads via executorch::extension::llm::load_tokenizer() (same as Parakeet/Voxtral).
Special tokens needed in C++: codec_bos_token_id, codec_eos_token_id, codec_pad_token_id. These are stored as constant_methods in the .pte.

5. CMake Updates
File: examples/models/qwen3-tts/CMakeLists.txt

Add:

tokenizers::tokenizers link target (for tiktoken C++ decoding)
extension_llm_runner (for load_tokenizer)
Source files: qwen3_tts_c_api.cpp
Remove: nlohmann/json dependency (no longer needed — metadata comes from constant_methods)
Follow Parakeet's CMakeLists.txt pattern.

6. Tests and Verification
Python tests:

test_dynamic_decoder_export.py: Verify the patched decoder exports with dynamic shapes and produces correct output at multiple input lengths (compare against bucketed output).
test_unified_export.py: Verify all 6 methods export into a single .pte, load, and execute independently.
test_code_predictor_baked.py: Verify the baked-in group embedding/head selection matches per-group module output.
C++ verification:

Build runner, run with precomputed codes (backward compat with --codes_path).
Run full text→audio pipeline with --text flag.
Compare output WAV against Python baseline for bit-exact parity on greedy decode.
Performance targets (8da4w XNNPACK, Apple Silicon CPU):

Talker: ~64 ms/step × 91 steps = ~5.8s
Code predictor: ~7 ms/step × 1365 steps = ~9.8s
Decoder: single run, dynamic shape, ~3s for 91 codes
Total: ~19s for 7.3s audio (2.6x realtime)
Model file: single .pte ~600-700 MB
Implementation Order
Fix decoder dynamic shapes — patch CausalConvNet, verify export works with Dim.AUTO
Create export wrapper modules — TokenEmbeddingExport, TextProjectionExport, TalkerPrefillExport, TalkerDecodeStepExport, CodePredictorStepExport, DynamicShapeDecoderExport
Write export_unified.py — multi-method export with per-component quantization
Verify .pte methods in Python — load and call each method, compare against eager model
Rewrite C++ runner — full text→audio pipeline with synthesize()
Add C API — qwen3_tts_c_api.h/.cpp following Parakeet pattern
Update CMake — link tokenizer, add C API source
End-to-end test — text→audio through C++ runner, compare against Python baseline
Critical Files
File	Action	Purpose
examples/models/qwen3-tts/model.py	Modify	Add DynamicShapeDecoderExport with patched conv padding
examples/models/qwen3-tts/export_unified.py	Create	Multi-method single-PTE export script
examples/models/qwen3-tts/qwen3_tts_runner.h	Rewrite	Multi-method runner with synthesize()
examples/models/qwen3-tts/qwen3_tts_runner.cpp	Rewrite	Full text→audio pipeline orchestration
examples/models/qwen3-tts/qwen3_tts_c_api.h	Create	C API for iOS/Android
examples/models/qwen3-tts/qwen3_tts_c_api.cpp	Create	C API implementation
examples/models/qwen3-tts/main.cpp	Modify	Add --text mode using unified runner
examples/models/qwen3-tts/CMakeLists.txt	Modify	Add tokenizer lib, C API source
examples/models/qwen3-tts/config/talker_config.json	Keep	Talker architecture config
examples/models/qwen3-tts/config/code_predictor_config.json	Keep	Code predictor architecture config
Existing Utilities to Reuse
executorch.exir.to_edge_transform_and_lower — multi-method export (same as Parakeet/Voxtral)
executorch.extension.llm.export.quantize.quantize_model_ — per-component quantization
executorch.extension.llm.load_tokenizer — C++ tokenizer loading (auto-detects format)
executorch.examples.models.llama.llama_transformer.construct_transformer — talker model construction (already used by export_talker.py)
executorch.backends.xnnpack.partition.XnnpackPartitioner / XnnpackDynamicallyQuantizedPartitioner — backend delegation
Parakeet C API pattern (parakeet_c_api.h) — thread-safe FFI wrapper
