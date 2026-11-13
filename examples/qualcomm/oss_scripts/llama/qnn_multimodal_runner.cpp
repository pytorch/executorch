/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run SmolVLM 500M, InternVL3 1B, GraniteSpeech3.3 2B
 * with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/chat_template.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <fstream>
#include <optional>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::llm::Audio;
using executorch::extension::llm::Image;
using ::executorch::extension::llm::make_audio_input;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using executorch::extension::llm::MultimodalInput;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

// Model paths
DEFINE_string(
    tok_embedding_path,
    "tok_embedding.pte",
    "Path to tok_embedding model serialized in flatbuffer format.");
DEFINE_string(
    encoder_path,
    "encoder.pte",
    "Path to vision encoder model serialized in flatbuffer format.");
DEFINE_string(
    decoder_path,
    "decoder.pte",
    "Path to decoder model serialized in flatbuffer format.");

// Tokenizer and output paths
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Executorch inference data output path.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed. For CI purpose.");
DEFINE_string(
    dump_logits_path,
    "",
    "If path is provided, program will dump all logits generated.");

// Model configuration
DEFINE_string(decoder_model_version, "llama3", "The decoder model version.");
DEFINE_string(
    prompt,
    "Describe this image:",
    "Text prompt for the multimodal model.");
DEFINE_string(
    tokenized_prompt,
    "",
    "This is an alternative of passing prompts. Users could provide this in a raw file, with tokens saved in uint64 format.");
DEFINE_string(
    audio_path,
    "",
    "Path to input audio file. For Audio-Language models, please specify audio path.");
DEFINE_string(
    image_path,
    "",
    "Path to input image file. For Vision-Language models, please specify image path.");
DEFINE_string(system_prompt, "", "System prompt for the model.");

// Generation parameters
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");

DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy use case between the application and device/co-processor associated with the backend.");

// Lookahead decoding parameters
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] Size of n-grams used in lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Number of future tokens to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Maximum number of speculations or candidate n-grams.");

// Speech output (LFM3-Omni)
DEFINE_string(depth_path, "", "Depth decoder .pte; enables speech output.");
DEFINE_string(embed_frame_path, "", "Speech frame embedding .pte.");
DEFINE_string(codec_path, "", "Speech codec decoder .pte.");
DEFINE_string(speech_output_path, "speech.wav", "Generated 24 kHz speech.");
DEFINE_string(
    generation_mode,
    "auto",
    "auto: the model picks text or speech; audio: open the answer with speech.");
DEFINE_double(
    audio_temperature,
    0.4,
    "Speech code temperature; omitted uses exported policy.");
DEFINE_double(
    audio_repetition_penalty,
    1.2,
    "Speech code penalty; omitted uses exported policy.");
DEFINE_int32(speech_seed, 0, "Seed for speech code sampling.");

// Execution parameters
DEFINE_int32(num_iters, 1, "Total number of iterations to run.");

std::vector<std::string> CollectPrompts(int argc, char** argv) {
  // Collect all prompts from command line, example usage:
  // --prompt "prompt1" --prompt "prompt2" --prompt "prompt3"
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++; // Skip the next argument
    }
  }
  return prompts;
}

void start_multimodal_runner(
    std::unique_ptr<executorch::extension::Module> encoder,
    std::unique_ptr<executorch::extension::Module> tok_embedding,
    std::unique_ptr<executorch::extension::Module> text_decoder,
    std::vector<std::string>& prompts) {
  ET_LOG(Info, "Starting multimodal runner");

  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;

  const bool lfm3 = FLAGS_decoder_model_version == "lfm3" ||
      FLAGS_decoder_model_version == "lfm3_speech";
  ET_CHECK_MSG(
      !lfm3 || (FLAGS_num_iters == 1 && prompts.size() == 1),
      "LFM3 multimodal generation currently supports one prompt and one iteration; "
      "use separate runner invocations for independent conversations");
  ET_CHECK_MSG(
      FLAGS_depth_path.empty()
          ? (FLAGS_embed_frame_path.empty() && FLAGS_codec_path.empty())
          : (!FLAGS_embed_frame_path.empty() && !FLAGS_codec_path.empty() &&
             !FLAGS_speech_output_path.empty() && FLAGS_eval_mode != 2),
      "Speech needs depth, embed_frame, codec and output paths, and no lookahead");

  // Create multimodal runner
  example::QNNMultimodalRunner runner(
      std::move(encoder),
      std::move(tok_embedding),
      std::move(text_decoder),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap);

  if (!FLAGS_depth_path.empty()) {
    auto load = [](const std::string& path) {
      return std::make_unique<executorch::extension::Module>(
          path,
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    };
    std::optional<float> audio_temperature;
    std::optional<float> audio_penalty;
    if (!gflags::GetCommandLineFlagInfoOrDie("audio_temperature").is_default) {
      audio_temperature = static_cast<float>(FLAGS_audio_temperature);
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("audio_repetition_penalty")
             .is_default) {
      audio_penalty = static_cast<float>(FLAGS_audio_repetition_penalty);
    }
    runner.set_speech(
        load(FLAGS_depth_path),
        load(FLAGS_embed_frame_path),
        load(FLAGS_codec_path),
        FLAGS_speech_output_path,
        audio_temperature,
        audio_penalty,
        static_cast<uint32_t>(FLAGS_speech_seed));
  }
  auto model_version = runner.get_model_version().get();

  // Prepare output buffer (similar to qnn_llama_runner.cpp)
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  executorch::extension::llm::GenerationConfig config;
  config.echo = true;
  config.ignore_eos = false;
  config.max_new_tokens = -1;
  config.warming = false;
  config.seq_len = FLAGS_seq_len;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.num_bos = 0;
  config.num_eos = 0;

  // 1. [Multimodal] Get raw files from input_list.txt
  std::vector<std::string> audio_raw_files;
  std::vector<std::string> image_raw_files;
  if (modality_of(model_version) == example::Modality::kAudio) {
    // LFM3-Omni also takes text-only prompts (e.g. text to speech).
    const auto* alm = std::get_if<example::AudioLanguageModel>(&model_version);
    const bool audio_optional =
        alm != nullptr && *alm == example::AudioLanguageModel::kLfm3OmniSpeech;
    ET_CHECK_MSG(
        audio_optional || !FLAGS_audio_path.empty(),
        "For ALM, please specify audio path.");
    if (!FLAGS_audio_path.empty()) {
      audio_raw_files = example::load_raw_files(FLAGS_audio_path.c_str());
    }
  }
  if (modality_of(model_version) == example::Modality::kVision) {
    ET_CHECK_MSG(
        lfm3 || !FLAGS_image_path.empty(),
        "For VLM, please specify image path.");
    if (!FLAGS_image_path.empty()) {
      image_raw_files = example::load_raw_files(FLAGS_image_path.c_str());
    }
  }

  // 2. Prepare messages for multi-turn simulation
  std::vector<Message> messages =
      prepare_messages(prompts, image_raw_files, audio_raw_files);

  // 3. Get expected input size/dtype for encoder
  Result<MethodMeta> method_meta = runner.get_encoder_method_meta();
  auto input_meta_result = method_meta->input_tensor_meta(0);
  std::vector<int32_t> expected_size(
      input_meta_result->sizes().begin(), input_meta_result->sizes().end());
  ScalarType expected_dtype = input_meta_result->scalar_type();

  // TODO: add use_tokenized_prompt for enable running static Llama models
  // inside LlamaDemo Android
  //  4. generate tokens & store inference output
  for (int i = 0; i < FLAGS_num_iters; i++) {
    for (size_t j = 0; j < messages.size(); ++j) {
      const auto& prompt = messages[j].text;
      const std::vector<std::string> files_path = messages[j].files_path;

      // 4.1 prepare modality input
      std::vector<MultimodalInput> inputs;
      // 4.1.1 prepare audio input
      if (modality_of(model_version) == example::Modality::kAudio) {
        for (const std::string& file_path : files_path) {
          Audio audio;
          example::load_audio(file_path, audio, expected_size, expected_dtype);
          inputs.emplace_back(make_audio_input(audio));
        }
      }
      // 4.1.2 prepare image input
      if (modality_of(model_version) == example::Modality::kVision) {
        for (const std::string& file_path : files_path) {
          Image image;
          example::load_image(file_path, image, expected_size, expected_dtype);
          inputs.emplace_back(make_image_input(image));
        }
      }

      // 4.2 apply chat template for text input
      std::string formatted_prompt =
          apply_chat_template(prompt, FLAGS_system_prompt, model_version);
      if (FLAGS_generation_mode == "audio") {
        formatted_prompt.append("<|audio_start|>");
      }
      inputs.emplace_back(make_text_input(formatted_prompt));

      // 4.3 dispatch inputs into correct order
      inputs = dispatch_inputs(inputs, formatted_prompt);

      // 4.4 generate text
      ET_CHECK_MSG(
          runner.generate(inputs, config, callback) ==
              executorch::runtime::Error::Ok,
          "Multimodal generation failed");
    }
  }
  fout.write(buf.data(), buf.size());
  fout.close();
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_generation_mode != "auto" && FLAGS_generation_mode != "audio") {
    ET_LOG(Error, "generation_mode must be auto or audio");
    return 1;
  }
  if (FLAGS_generation_mode == "audio" &&
      (FLAGS_depth_path.empty() ||
       FLAGS_decoder_model_version != "lfm3_speech")) {
    ET_LOG(
        Error,
        "Audio generation requires an LFM3 speech decoder and depth model");
    return 1;
  }
  if (prompts.empty()) {
    prompts.push_back(FLAGS_prompt);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Only provide prompt or tokenized_input but not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(
        false, "Only TokenGenerator(kv) mode is supported to dump all logits.");
  }

  // Load encoder
  ET_LOG(Info, "Load Encoder: %s", FLAGS_encoder_path.c_str());
  std::unique_ptr<executorch::extension::Module> encoder =
      std::make_unique<executorch::extension::Module>(
          FLAGS_encoder_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Load token embedding
  ET_LOG(Info, "Load Token Embedding: %s", FLAGS_tok_embedding_path.c_str());
  std::unique_ptr<executorch::extension::Module> tok_embedding =
      std::make_unique<executorch::extension::Module>(
          FLAGS_tok_embedding_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Load text decoder
  ET_LOG(Info, "Load Text Decoder: %s", FLAGS_decoder_path.c_str());
  std::unique_ptr<executorch::extension::Module> text_decoder =
      std::make_unique<executorch::extension::Module>(
          FLAGS_decoder_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Start runner
  start_multimodal_runner(
      std::move(encoder),
      std::move(tok_embedding),
      std::move(text_decoder),
      prompts);

  return 0;
}
