# LFM3 on Qualcomm HTP

Static LFM3 models use the shared decoder in `examples/qualcomm/oss_scripts/llama/model/static_llama.py` and the converter/config in `examples/models/lfm3`. Model keys follow `lfm3-<size>[_variant]`.

| `--decoder_model` | Hugging Face | Modality |
|---|---|---|
| `lfm3-1_7b` | LiquidAI/LFM3-1.7B | text |
| `lfm3-1_7b_base` | LiquidAI/LFM3-1.7B-Base | text (base) |
| `lfm3-2b_vl` | LiquidAI/LFM3-VL-2B | image + text |
| `lfm3-2b_omni` | LiquidAI/LFM3-Omni-2B | image + text |
| `lfm3-2b_omni_speech` | LiquidAI/LFM3-Omni-2B | speech + text in, text and speech out |

Short-convolution layers carry their rolling state as graph IO after the V caches; the runners'
`ConvStateManager` keeps the last `conv_L_cache - 1` valid rows between steps. Lookahead decoding,
attention sink and sharding are not supported for these models.

## Text
```
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m SM8750 \
  --decoder_model lfm3-1_7b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 \
  --temperature 0 --calib_tasks wikitext --calib_limit 1 \
  --prompt "What is the capital of France? Answer in one sentence."
```

## Image + text
The vision tower and projector run in fp16 on a fixed 448x448 input (196 image tokens); images are
resized to 448x448. Calibration samples must use the model's image markup,
`<|vision_start|><|image|><|vision_end|>`, so image embeddings are spliced in during calibration:
```json
[{"files": ["image.png"], "messages": [
  {"role": "user", "content": "<|vision_start|><|image|><|vision_end|>Describe this image."},
  {"role": "assistant", "content": "..."}]}]
```
```
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m SM8750 \
  --decoder_model lfm3-2b_omni --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 \
  --temperature 0 --calib_tasks wikitext --calib_limit 1 --calib_samples calib.json \
  --image_path image.png --prompt "<image>What shapes and colors are in this image?"
```

## Speech in, text and speech out (LFM3-Omni)
`lfm3-2b_omni_speech` exports the decoder with its final hidden state as an extra output (right after
the logits) plus the speech encoder (fp16, a fixed 10.24 s window of 1024 mel frames; the input's
last row marks the valid frames and the runner keeps `ceil(frames / 8)` audio tokens). Speech output
also needs the depth decoder, the speech-frame embedding and the codec (fp16):
```
python examples/qualcomm/oss_scripts/lfm3/capture_speech_calibration.py \
  --audio question.wav -o calib_embeds.pt   # HF remote code, see the script
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m SM8750 \
  --decoder_model lfm3-2b_omni_speech --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 \
  --temperature 0 --calib_tasks wikitext --calib_limit 1 --calib_samples calib_speech.json \
  --calib_embeddings calib_embeds.pt --audio_path question.wav --compile_only -a artifacts
python examples/qualcomm/oss_scripts/lfm3/export_speech_components.py -a artifacts
```
`--calib_embeddings` is required for a fresh quantized speech-output decoder export. It adds the decoder
input sequences HF feeds while speaking (summed codebook embeddings of generated frames), which
text and audio samples do not produce. `--pre_gen_pte` reuse and `--use_fp16` export do not
run PTQ and are exempt from this calibration gate. The existing speech runner still requires
quantized uint16 decoder outputs; the exemption does not add FP16 decoder support. Calibration samples
mark audio as `<|vision_start|><|audio|><|vision_end|>`.

Run with `qnn_multimodal_runner --decoder_model_version lfm3_speech --encoder_path audio_encoder_qnn.pte
--depth_path depth.pte --embed_frame_path embed_frame.pte --codec_path codec.pte
--speech_output_path answer.wav` and `--audio_path` (raw `[1, 129, 1024]` float features from
`preprocess.audio_features`) and/or a text prompt. `--generation_mode audio` opens the answer with
speech (text to speech); otherwise the model picks text, speech, or interleaved text and speech.
Speech text slots, including the first prefill slot, use the policy exported in `depth.pte` from
`generation_config.json`: `do_sample`, `temperature`, `top_k`, `top_p`, `min_p`,
`repetition_penalty`, and `text_repetition_window`. An explicit
`export_speech_components.py --generation_config settings.json` replaces that file; it does not
infer settings from a pipeline. The replacement JSON must explicitly set `do_sample`,
`temperature`, `top_k`, and `repetition_penalty`. The checkpoint at revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67` has greedy text and penalty **1.0**, not 1.1.
Its generated-only text history defaults to 32 tokens; audio history is 64 nonterminal frames
per codebook, not full history. A window of zero disables that history. The audio top-k and
history window also come from the exported policy; the existing runtime
`--audio_temperature 0.4 --audio_repetition_penalty 1.2` arguments override the exported audio
values only when those flags are explicitly supplied. Omitting them preserves the serialized
values; an explicit `--audio_temperature 0` selects greedy codes. `--temperature` is not a
speech-text override.

These policies follow that revision's `modeling_lfm3_omni.py::_sample` and `_generate`.
An EOS in **any** codebook canonicalizes the entire frame to EOS, feeds its embedding back,
excludes it from codec output/history, and returns to text. A later `<|audio_start|>` can open
another speech segment; history is not reset on reentry. This is reference behavior, not a
codebook-zero-only stop rule. Frames of successive segments are currently concatenated for
codec decoding without an explicit segment boundary.

The codec replays overlapping windows with enough left context for all temporal layers and right context for iSTFT synthesis. Its valid-frame input excludes final padding from both synthesis and normalization. Re-export `depth.pte`, `embed_frame.pte` and `codec.pte` together; older component contracts are rejected at load.

`capture_speech_calibration.py` runs Hugging Face remote modeling code on the host only and is not part of CI.

Known limitations: LFM3 multimodal generation is single-turn and accepts one iteration per
invocation. Speech text sampling does not report sampling-time/logits statistics. The supported
sampling fields above are not a complete implementation of every HF generation option (for
example, beam search, stop strings, and custom processors are not implemented). Re-export
`depth.pte` to supply sampling metadata; missing/obsolete policy metadata is rejected. Sampling
uses dequantized uint16 text logits and float speech logits, with a C++ RNG rather than PyTorch's
RNG: identical seeds do not imply identical frame orders or samples. Long speech outputs
(about 9 s and more) have been reported to drop words on HTP. Policy tests alone do not establish
the root cause or fix that observation; full-model fp32/QDQ/HTP replay and ASR checks remain
necessary. Codec continuity tests against the static host decoder do not establish long-output
parity with the HF codec or continuity across speech reentry.
