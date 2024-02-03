/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama2/runner/runner.h>

#include <ctime>

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/examples/models/llama2/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>

namespace torch {
namespace executor {

Runner::Runner(const char* model_path, const char* tokenizer_path) {
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
  const auto method_names = module_->method_names();
  ET_CHECK_MSG(
      method_names.ok(),
      "Failed to read method names from model: %s",
      model_path);
  model_methods_ = method_names.get();
  vocab_size_ = getMetadataHelper<int64_t>("get_vocab_size", 32000);
  bos_id_ = getMetadataHelper<int64_t>("get_bos_id", 1);
  eos_id_ = getMetadataHelper<int64_t>("get_eos_id", 2);
  n_bos_ = getMetadataHelper<int64_t>("get_n_bos", 1);
  n_eos_ = getMetadataHelper<int64_t>("get_n_eos", 1);
  max_seq_len_ = getMetadataHelper<int64_t>("get_max_seq_len", 128);

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

template <typename T>
T Runner::getMetadataHelper(std::string method_name, T default_val) {
  T res = default_val;
  if (model_methods_.count(method_name)) {
    Result<std::vector<EValue>> outputs = module_->execute(method_name);
    if (outputs.ok()) {
      std::vector<EValue> outs = outputs.get();
      if (outs.size() > 0) {
        res = outs[0].to<T>();
      }
    }
  } else {
    ET_LOG(
        Info,
        "The model does not contain %s method, using default value %ld",
        method_name.c_str(),
        (int64_t)default_val);
  }
  ET_LOG(Info, "%s: %ld", method_name.c_str(), (int64_t)res);
  return res;
}

template <typename T>
int32_t Runner::logitsToToken(
    const exec_aten::Tensor& logits_tensor,
    int64_t pos,
    T _) {
  (void)_;
  T* logits = logits_tensor.mutable_data_ptr<T>();

  // Since the logits are for all tokens, get the last token probabilities
  T* logits_last = logits + pos * tokenizer_->vocab_size();
  return sampler_->sample(logits_last);
}

Error Runner::generate(
    const char* prompt,
    bool eos,
    std::function<void(const std::string&)> callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(prompt != nullptr, "Prompt cannot be null");

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  // max # of prompt tokens: len(prompt) + '\0', ?BOS, ?EOS
  int* prompt_tokens = new int[strlen(prompt) + 1 + n_bos_ + n_eos_];

  tokenizer_->encode(
      prompt, n_bos_, eos ? n_eos_ : 0, prompt_tokens, &num_prompt_tokens);
  for (int i = 0; i < num_prompt_tokens; i++) {
    ET_LOG(Info, "prompt_tokens[%d]: %d", i, prompt_tokens[i]);
  }
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

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
    exec_aten::Tensor logits_tensor = outputs.at(2).toTensor();

    switch (logits_tensor.scalar_type()) {
      case ScalarType::Float: {
        next_tok = logitsToToken<float>(logits_tensor, pos, 0);
        break;
      }
      case ScalarType::Half: {
        next_tok = logitsToToken<exec_aten::Half>(logits_tensor, pos, 0);
        break;
      }
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported dtype output %hhd",
            logits_tensor.scalar_type());
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

    if (callback) {
      callback(piece);
    }

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

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
    std::string method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
    std::string method_name,
    bool default_val);
} // namespace executor
} // namespace torch
