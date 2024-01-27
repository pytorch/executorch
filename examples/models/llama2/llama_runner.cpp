/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama2/llama_runner.h>
#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif
using torch::executor::util::MmapDataLoader;

namespace torch {
namespace executor {

LlamaRunner::LlamaRunner(const char* model_path, const char* tokenizer_path) {
  // Constants definition
  float temperature = 0.8f;
  float topp = 0.9f;
  unsigned long long rng_seed =
      (unsigned int)time(nullptr); // seed rng with time by default
  // Create module
  module_ = std::make_unique<Module>(
      model_path, Module::MlockConfig::UseMlockIgnoreErrors);

  // Read out metadata: vocab_size (expected by the model), BOS, EOS, n_BOS,
  // n_EOS max_seq_len from the model
  ET_LOG(Info, "Reading metadata from model");
  const auto method_names = module_->methodNames();
  ET_CHECK_MSG(
      method_names.ok(),
      "Failed to read method names from model: %s",
      model_path);
  auto metadata = readMetadata(method_names.get());
  ET_CHECK_MSG(metadata.size() == 6, "Invalid metadata size");
  vocab_size_ = metadata[0];
  bos_id_ = metadata[1];
  eos_id_ = metadata[2];
  n_bos_ = metadata[3];
  n_eos_ = metadata[4];
  max_seq_len_ = metadata[5];

  // Load tokenizer
  tokenizer_ = std::make_unique<Tokenizer>(vocab_size_, bos_id_, eos_id_);
  tokenizer_->load(tokenizer_path);
  if (tokenizer_->bos_tok() != bos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's BOS id %d does not match model's BOS id %d, will override tokenizer's BOS.",
        tokenizer_->bos_tok(),
        bos_id_);
  }
  if (tokenizer_->eos_tok() != eos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's EOS id %d does not match model's EOS id %d, will override tokenizer's EOS.",
        tokenizer_->eos_tok(),
        eos_id_);
  }
  // Create sampler
  sampler_ =
      std::make_unique<Sampler>(vocab_size_, temperature, topp, rng_seed);
}

std::vector<int32_t> LlamaRunner::readMetadata(
    std::unordered_set<std::string> model_methods) {
  std::vector<std::string> methods = {
      "get_vocab_size",
      "get_bos_id",
      "get_eos_id",
      "get_n_bos",
      "get_n_eos",
      "get_max_seq_len"};
  std::vector<int32_t> default_values = {32000, 1, 2, 1, 1, 128};
  std::vector<int32_t> result;
  for (int i = 0; i < methods.size(); ++i) {
    int32_t res = default_values[i];
    if (model_methods.count(methods[i])) {
      Result<std::vector<EValue>> outputs = module_->execute(methods[i]);
      if (outputs.ok()) {
        std::vector<EValue> outs = outputs.get();
        if (outs.size() > 0) {
          res = outs[0].toInt();
        }
      }
    } else {
      ET_LOG(
          Info,
          "The model does not contain %s method, using default value %d",
          methods[i].c_str(),
          default_values[i]);
    }
    ET_LOG(Info, "%s: %d", methods[i].c_str(), res);
    result.push_back(res);
  }
  return result;
}

Error LlamaRunner::generate(const char* prompt, bool eos) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(prompt != nullptr, "Prompt cannot be null");

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  // max # of prompt tokens: len(prompt) + '\0', ?BOS, ?EOS
  int* prompt_tokens = new int[strlen(prompt) + 1 + n_bos_ + n_eos_];

  tokenizer_->encode(
      prompt, n_bos_, eos ? n_eos_ : 0, prompt_tokens, &num_prompt_tokens);

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");

  // start the main loop
  long start =
      0; // used to time our code, only initialized after first iteration
  int next; // will store the next token in the sequence
  int pos = num_prompt_tokens - 1; // position in the sequence
  int token = prompt_tokens[pos]; // prefill starts from 0 to num_prompt_tokens
  int eos_counter = 0; // counter to capture EOS
  void* data =
      malloc(max_seq_len_ * sizeof(int64_t)); // allocate space for the tokens
  // copy prompt tokens into data
  for (int i = 0; i < num_prompt_tokens; ++i) {
    ((int64_t*)data)[i] = prompt_tokens[i];
    if (i > 0) {
      printf(
          "%s",
          ET_UNWRAP(
              tokenizer_->decode(prompt_tokens[i - 1], prompt_tokens[i])));
    }
  }
  // create a 1xN int tensor with next as value
  exec_aten::SizesType sizes[2]{1, pos};
  exec_aten::DimOrderType dim_order[2]{0, 1};

  while (pos < max_seq_len_) {
    // ET_LOG(Info, "Generating step %d...", pos);
    // set the current token in the tensor
    ((int64_t*)data)[pos] = token;
    sizes[1] = pos + 1;
#ifdef USE_ATEN_LIB
    exec_aten::Tensor tensor =
        torch::from_blob(data, /*dim*/ sizes, torch::kLong);
#else
    exec_aten::TensorImpl tensor_impl(
        ScalarType::Long, /*dim*/ 2, sizes, data, dim_order);
    exec_aten::Tensor tensor(&tensor_impl);
#endif
    Result<std::vector<EValue>> outputs_res =
        module_->forward({EValue(tensor)});
    ET_CHECK_MSG(
        outputs_res.ok(),
        "Execution of method forward failed with status 0x%" PRIx32,
        outputs_res.error());
    // ET_LOG(Info, "Model executed successfully.");

    std::vector<EValue> outputs = outputs_res.get();
    // Check the outputs.
    ET_CHECK_MSG(
        outputs.size() > 0,
        "Expecting output to have at least one evalue. Got %zu",
        outputs.size());

    int32_t next_tok;
    switch (outputs[0].toTensor().scalar_type()) {
      case ScalarType::Float: {
        float* logits = outputs[0].toTensor().mutable_data_ptr<float>();

        // Since the logits are for all tokens, get the last token probabilities
        float* logits_last = logits + pos * tokenizer_->vocab_size();
        next_tok = sampler_->sample(logits_last);
        break;
      }
      case ScalarType::Half: {
        exec_aten::Half* half_logits =
            outputs[0].toTensor().mutable_data_ptr<exec_aten::Half>();
        exec_aten::Half* logits_last =
            half_logits + pos * tokenizer_->vocab_size();
        next_tok = sampler_->sample(logits_last);
        break;
      }
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported dtype output %hhd",
            outputs[0].toTensor().scalar_type());
    }

    // debug
    // torch::Tensor t = torch::from_blob(
    //     (void*)logits, {1, num_prompt_tokens, 32000}, torch::kFloat);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // prefill, force the next token to be the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = next_tok;
    }
    // ET_LOG(Info, "Output saved, next = %d", next);
    pos++;

    // print the token as string, decode it with the Tokenizer object
    auto piece_res = tokenizer_->decode(token, next);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get();

    // same as printf("%s", piece), but skips "unsafe" bytes
    util::safe_printf(piece);
    fflush(stdout);

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens && next == eos_id_) {
      eos_counter++;
      if (eos_counter == n_eos_) {
        ET_LOG(Info, "Reached to the end of generation");
        break;
      }
    } else {
      eos_counter = 0;
    }

    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = util::time_in_ms();
    }
  }
  printf("\n");

  if (pos == max_seq_len_) {
    ET_LOG(Info, "Maximum sequence length reached!");
  }
  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (pos >= 1) {
    long end = util::time_in_ms();
    ET_LOG(
        Info, "Achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  delete[] prompt_tokens;
  free(data);
  return Error::Ok;
}

LlamaRunner::~LlamaRunner() {}
} // namespace executor
} // namespace torch
