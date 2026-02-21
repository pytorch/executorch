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
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/chat_template.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <fstream>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::llm::Image;
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

template <typename T>
void start_multimodal_runner(
    std::unique_ptr<executorch::extension::Module> encoder,
    std::unique_ptr<executorch::extension::Module> tok_embedding,
    std::unique_ptr<executorch::extension::Module> text_decoder,
    std::vector<std::string>& prompts) {
  ET_LOG(Info, "Starting multimodal runner");

  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;

  // Create multimodal runner
  example::MultimodalRunner<T> runner(
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

  auto model_version = runner.get_model_version().get();

  if (modality_of(model_version) == example::Modality::kVision) {
    ET_CHECK_MSG(
        !FLAGS_image_path.empty(),
        "For VLM models, please specify image path.");
  }

  // Prepare output buffer (similar to qnn_llama_runner.cpp)
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  executorch::extension::llm::GenerationConfig config{
      true,
      false,
      -1,
      false,
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  // 1. [Multi-modality] Get raw files from input_list.txt
  std::vector<std::string> raw_files =
      example::load_raw_files(FLAGS_image_path.c_str());

  // 2. Prepare messages for multi-turn simulation
  std::vector<Message> messages = prepare_messages(prompts, raw_files);

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

      // 4.1 prepare image input
      std::vector<MultimodalInput> inputs;
      if (modality_of(model_version) == example::Modality::kVision) {
        for (const std::string& file_path : files_path) {
          Image image;
          example::load_image(file_path, image, expected_size, expected_dtype);
          inputs.emplace_back(make_image_input(image));
        }
      }

      // 4.2 prepare prompt input
      std::string formatted_prompt =
          apply_chat_template(prompt, FLAGS_system_prompt, model_version);
      inputs.emplace_back(make_text_input(formatted_prompt));

      // 4.3 generate text
      runner.generate_from_prompt_or_file(
          inputs, use_tokenized_prompt, config, callback);
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

  // Using 8bit as default since this meta is introduced with 16bit kv io
  // support and older models only have 8bit kv io.
  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (text_decoder->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        text_decoder->get("get_kv_io_bit_width")
            .get()
            .toScalar()
            .to<int64_t>());
  }
  // Start runner with appropriate KV bitwidth
  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    start_multimodal_runner<uint8_t>(
        std::move(encoder),
        std::move(tok_embedding),
        std::move(text_decoder),
        prompts);
  } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    start_multimodal_runner<uint16_t>(
        std::move(encoder),
        std::move(tok_embedding),
        std::move(text_decoder),
        prompts);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(kv_bitwidth));
  }

  return 0;
}
