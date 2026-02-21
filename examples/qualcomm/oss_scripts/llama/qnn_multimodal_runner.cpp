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
 * This tool can run SmolVLM 500M, InternVL3 1B
 * with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <fstream>
#include <vector>

// Model paths
DEFINE_string(
    embedding_path,
    "embedding.pte",
    "Path to embedding model serialized in flatbuffer format.");
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
    image_path,
    "",
    "Path to input image file. If empty, text-only mode is used.");
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

/**
 * Special tokens structure for different models
 */
struct SpecialTokens {
  std::string image_token;
  std::string global_img;
  std::string fake_wrap_start;
  std::string fake_wrap_end;
};

/**
 * Get special tokens based on decoder model version
 */
SpecialTokens get_special_tokens(
    example::MultimodalDecoderModelVersion decoder_model_version) {
  SpecialTokens tokens;

  switch (decoder_model_version) {
    case example::MultimodalDecoderModelVersion::
        kSmolvlm: // smolvlm_500m_instruct
      tokens.image_token = "<image>";
      tokens.global_img = "<global-img>";
      tokens.fake_wrap_start = "<fake_token_around_image>";
      tokens.fake_wrap_end = "<fake_token_around_image>";
      break;
    case example::MultimodalDecoderModelVersion::kInternvl3: // internvl3_1b
      tokens.image_token = "<IMG_CONTEXT>";
      tokens.global_img = "";
      tokens.fake_wrap_start = "<img>";
      tokens.fake_wrap_end = "</img>";
      break;
    default:
      break;
  }

  return tokens;
}

/**
 * Prepare multimodal token IDs by expanding image tokens
 * This implements the logic from prepare_multimodal_token_ids in Python
 */
std::string prepare_multimodal_prompt(
    const std::string& prompt,
    int image_seq_len,
    const SpecialTokens& specials) {
  // Create image prompt with repeated image tokens
  std::string image_prompt = specials.fake_wrap_start;
  image_prompt += specials.global_img;
  for (int i = 0; i < image_seq_len; ++i) {
    image_prompt += specials.image_token;
  }
  image_prompt += specials.fake_wrap_end;

  // Replace single image token with expanded version
  size_t pos = 0;
  std::string expanded = prompt;
  while ((pos = expanded.find(specials.image_token, pos)) !=
         std::string::npos) {
    expanded.replace(pos, specials.image_token.size(), image_prompt);
    pos += image_prompt.size();
  }
  ET_LOG(Info, "Prompt after expanding image token: %s", expanded.c_str());

  return expanded;
}

/**
 * Format prompt based on model version with multimodal token expansion
 */
std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::MultimodalDecoderModelVersion decoder_model_version,
    int32_t img_seq_len = 0) {
  std::string formatted_prompt;

  // Get special tokens for this model
  SpecialTokens specials = get_special_tokens(decoder_model_version);

  switch (decoder_model_version) {
    case example::MultimodalDecoderModelVersion::kSmolvlm:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|im_start|>User:");
      formatted_prompt.append(specials.image_token);
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_utterance>\nAssistant:");
      break;
    case example::MultimodalDecoderModelVersion::kInternvl3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system<|im_end|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>");
      }
      formatted_prompt.append("<|im_start|>user:\n");
      formatted_prompt.append(specials.image_token);
      formatted_prompt.append("\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>assistant\n");
      break;
    default:
      ET_CHECK_MSG(false, "unsupported VLM version");
      break;
  }

  // Expand image tokens
  formatted_prompt =
      prepare_multimodal_prompt(formatted_prompt, img_seq_len, specials);

  return formatted_prompt;
}

template <typename T>
void start_multimodal_runner(
    std::unique_ptr<example::EncoderRunner> encoder_runner,
    std::unique_ptr<executorch::extension::Module> module,
    std::unique_ptr<executorch::extension::Module> embedding,
    std::vector<std::string>& prompts) {
  ET_LOG(Info, "Starting multimodal runner");

  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;

  // Load image, run encoder forward pass, and set image hidden states if
  // provided
  bool has_image = !FLAGS_image_path.empty();

  // Load encoder
  if (encoder_runner->load() != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load encoder");
    return;
  }

  // Encode image from file
  auto encode_result =
      encoder_runner->encode_from_file(FLAGS_image_path.c_str());
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode image");
    return;
  }

  auto image_hidden_states = encode_result.get();

  // Create multimodal runner
  example::MultimodalRunner<T> runner(
      std::move(module),
      std::move(embedding),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_decoder_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      std::make_unique<executorch::aten::Tensor>(image_hidden_states));

  auto decoder_model_version = runner.get_decoder_model_version();

  // Prepare output buffer (similar to qnn_llama_runner.cpp)
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());

  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };

  // Configure generation
  executorch::extension::llm::GenerationConfig config{
      true,
      false,
      -1,
      false,
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  // Get image sequence length from encoder
  int32_t img_seq_len = encoder_runner->get_image_seq_len();
  if (use_tokenized_prompt) {
    runner.generate_from_prompt_or_file(
        FLAGS_tokenizer_path.c_str(), use_tokenized_prompt, config, callback);
  } else {
    // generate tokens & store inference output
    for (int i = 0; i < FLAGS_num_iters; i++) {
      for (size_t j = 0; j < prompts.size(); ++j) {
        const auto& prompt = prompts[j];
        std::string formatted_prompt;
        formatted_prompt = get_formatted_prompt(
            prompt,
            FLAGS_system_prompt,
            decoder_model_version.get(),
            img_seq_len);
        runner.generate_from_prompt_or_file(
            formatted_prompt.c_str(), use_tokenized_prompt, config, callback);
      }
    }
  }
  fout.write(buf.data(), buf.size());
  fout.close();
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Only provide prompt or tokenized_input but not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(
        false, "Only TokenGenerator(kv) mode is supported to dump all logits.");
  }
  ET_LOG(Info, "Embedding: %s", FLAGS_embedding_path.c_str());
  ET_LOG(Info, "Encoder: %s", FLAGS_encoder_path.c_str());
  ET_LOG(Info, "Decoder: %s", FLAGS_decoder_path.c_str());

  // Create encoder runner
  std::unique_ptr<example::EncoderRunner> encoder_runner =
      std::make_unique<example::EncoderRunner>(FLAGS_encoder_path.c_str());

  // load embedding
  std::unique_ptr<executorch::extension::Module> embedding =
      std::make_unique<executorch::extension::Module>(
          FLAGS_embedding_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

  // load decoder
  std::unique_ptr<executorch::extension::Module> module =
      std::make_unique<executorch::extension::Module>(
          FLAGS_decoder_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Using 8bit as default since this meta is introduced with 16bit kv io
  // support and older models only have 8bit kv io.
  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (module->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }
  // Start runner with appropriate KV bitwidth
  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    start_multimodal_runner<uint8_t>(
        std::move(encoder_runner),
        std::move(module),
        std::move(embedding),
        prompts);
  } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    start_multimodal_runner<uint16_t>(
        std::move(encoder_runner),
        std::move(module),
        std::move(embedding),
        prompts);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(kv_bitwidth));
  }

  return 0;
}
