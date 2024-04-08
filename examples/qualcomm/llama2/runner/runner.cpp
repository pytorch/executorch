/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama2/tokenizer/bpe_tokenizer.h>
#include <executorch/examples/qualcomm/llama2/runner/runner.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/managed_tensor.h>

#include <ctime>
#include <memory>
#include <sstream>

#include <executorch/examples/models/llama2/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch {
namespace executor {

namespace {
static constexpr auto kTopp = 0.9f;
void printReport(const Runner::Stats& stats);
std::string statsToJsonString(const Runner::Stats& stats);
} // namespace

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    : module_(std::make_unique<Module>(
          model_path,
          Module::MlockConfig::UseMlockIgnoreErrors)),
      tokenizer_path_(tokenizer_path),
      temperature_(temperature) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && sampler_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_.model_load_start_ms = util::time_in_ms();
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("forward"));

  // Read out metadata from the model
  ET_LOG(Info, "Reading metadata from model");
  const auto method_names = module_->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
  model_methods_ = method_names.get();
  vocab_size_ = getMetadataHelper<int64_t>("get_vocab_size", 32000);
  bos_id_ = getMetadataHelper<int64_t>("get_bos_id", 1);
  eos_id_ = getMetadataHelper<int64_t>("get_eos_id", 2);
  n_bos_ = getMetadataHelper<int64_t>("get_n_bos", 1);
  n_eos_ = getMetadataHelper<int64_t>("get_n_eos", 1);
  max_seq_len_ = getMetadataHelper<int64_t>("get_max_seq_len", 128);

  // Load tokenizer
  tokenizer_ = std::make_unique<BPETokenizer>(vocab_size_, bos_id_, eos_id_);
  tokenizer_->load(tokenizer_path_);
  if (tokenizer_->bos_tok() != bos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's BOS id %lu does not match model's BOS id %d, will override tokenizer's BOS.",
        tokenizer_->bos_tok(),
        bos_id_);
  }
  if (tokenizer_->eos_tok() != eos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's EOS id %lu does not match model's EOS id %d, will override tokenizer's EOS.",
        tokenizer_->eos_tok(),
        eos_id_);
  }
  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));
  stats_.model_load_end_ms = util::time_in_ms();

  return Error::Ok;
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
        "The model does not contain %s method, using default value %lld",
        method_name.c_str(),
        (long long)default_val);
  }
  ET_LOG(Info, "%s: %lld", method_name.c_str(), (long long)res);
  return res;
}

template <typename T>
int32_t Runner::logitsToToken(const exec_aten::Tensor& logits_tensor) {
  T* logits = logits_tensor.mutable_data_ptr<T>();

  // Since the logits are for all tokens, get the last token probabilities
  T* logits_last = logits;
  return sampler_->sample(logits_last);
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> Runner::run_model_step(
    int64_t input_token,
    Tensor& token,
    Tensor& start_pos,
    std::vector<Tensor>& input_tensors) {
  token.mutable_data_ptr<int32_t>()[0] = input_token;
  // inputs:[tokens, start_pos, atten_mask, kv_mask, k_cache, v_cache]
  std::vector<EValue> inputs = {token, start_pos};
  inputs.insert(inputs.end(), input_tensors.begin(), input_tensors.end());

  Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());

  // Bump start_pos by 1
  start_pos.mutable_data_ptr<int32_t>()[0]++;
  return outputs_res.get()[1].toTensor();
}

// TODO: add overloaded method for on-device tokenize
Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::vector<ManagedTensor>& managed_inputs,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  ET_CHECK_MSG(is_loaded(), "Please invoke load method first");

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.
  stats_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in static_llama.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  int32_t pos = 0, prev_token, cur_token = prompt_tokens[0];
  std::vector<int32_t> token_data = {1};
  std::vector<exec_aten::SizesType> token_shape = {1, 1};

  std::vector<int32_t> start_pos_data = {0};
  std::vector<exec_aten::SizesType> start_pos_shape = {1, 1};

  std::vector<float> logits_data(vocab_size_);
  std::vector<exec_aten::SizesType> logits_data_shape = {1, vocab_size_};

  // initialize tensor wrappers
  ManagedTensor managed_token(
      token_data.data(), 128, token_shape, ScalarType::Int);
  ManagedTensor managed_pos_id(
      start_pos_data.data(), 128, start_pos_shape, ScalarType::Int);
  ManagedTensor managed_logits(
      logits_data.data(), 128, logits_data_shape, ScalarType::Float);

  Tensor logits = managed_logits.get_aliasing_tensor();
  Tensor token = managed_token.get_aliasing_tensor();
  Tensor start_pos = managed_pos_id.get_aliasing_tensor();

  // TODO: investigate why kv_mask was duplicated in the output
  //       current output: [kv_mask, logits, k_cache, v_cache, kv_mask]
  //       change following indexes back when issue got resolved
  std::vector<Tensor> inputs;
  for (int i = 0; i < managed_inputs.size(); ++i) {
    inputs.push_back(managed_inputs[i].get_aliasing_tensor());
    ET_CHECK_MSG(
        module_->set_output_data_ptr(inputs.back(), i + 2) == Error::Ok,
        "Failed to set output tensor");
  }
  ET_CHECK_MSG(
      module_->set_output_data_ptr(logits, 1) == Error::Ok,
      "Failed to set output tensor - logits");

  // Start consuming user's prompts and generating new tokens
  std::string final_output;
  while (pos < seq_len - 1) {
    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(cur_token, token, start_pos, inputs);

    if (pos == num_prompt_tokens) {
      stats_.first_token_ms = util::time_in_ms();
    } else if (pos == num_prompt_tokens - 1) {
      stats_.prompt_eval_end_ms = util::time_in_ms();
    }

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();
    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logitsToToken<float>(logits_tensor);
    stats_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // prefill, force the next token to be the next prompt token
      cur_token = prompt_tokens[pos + 1];
    }
    pos++;

    // print the token as string, decode it with the Tokenizer object
    auto piece_res = tokenizer_->decode(prev_token, cur_token);
    ET_CHECK(piece_res.ok());

    if (token_callback) {
      token_callback(piece_res.get());
    }

    if (shouldStop_) {
      break;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens && cur_token == eos_id_) {
      ET_LOG(Info, "Reached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = util::time_in_ms();

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = pos - num_prompt_tokens;
  printReport(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

namespace {
void printReport(const Runner::Stats& stats) {
  printf("PyTorchObserver %s\n", statsToJsonString(stats).c_str());

  ET_LOG(
      Info,
      "\tPrompt Tokens: %" PRIu64 "    Generated Tokens: %" PRIu64,
      stats.num_prompt_tokens,
      stats.num_generated_tokens);

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t%f (seconds)",
      ((double)(stats.model_load_end_ms - stats.model_load_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));
  double inference_time_ms =
      (double)(stats.inference_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\tTotal inference time:\t\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      inference_time_ms / stats.SCALING_FACTOR_UNITS_PER_SECOND,

      (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);
  double prompt_eval_time =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\t\tPrompt evaluation:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      prompt_eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      (stats.num_prompt_tokens) / prompt_eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  double eval_time =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  ET_LOG(
      Info,
      "\t\tGenerated %" PRIu64
      " tokens:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      stats.num_generated_tokens,
      eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      stats.num_generated_tokens / eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  // Time to first token is measured from the start of inference, excluding
  // model load time.
  ET_LOG(
      Info,
      "\tTime to first generated token:\t%f (seconds)",
      ((double)(stats.first_token_ms - stats.inference_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tSampling time over %" PRIu64 " tokens:\t%f (seconds)",
      stats.num_prompt_tokens + stats.num_generated_tokens,
      (double)stats.aggregate_sampling_time_ms /
          stats.SCALING_FACTOR_UNITS_PER_SECOND);
}

std::string statsToJsonString(const Runner::Stats& stats) {
  std::stringstream ss;
  ss << "{\"prompt_tokens\":" << stats.num_prompt_tokens << ","
     << "\"generated_tokens\":" << stats.num_generated_tokens << ","
     << "\"model_load_start_ms\":" << stats.model_load_start_ms << ","
     << "\"model_load_end_ms\":" << stats.model_load_end_ms << ","
     << "\"inference_start_ms\":" << stats.inference_start_ms << ","
     << "\"inference_end_ms\":" << stats.inference_end_ms << ","
     << "\"prompt_eval_end_ms\":" << stats.prompt_eval_end_ms << ","
     << "\"first_token_ms\":" << stats.first_token_ms << ","
     << "\"aggregate_sampling_time_ms\":" << stats.aggregate_sampling_time_ms
     << ","
     << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << stats.SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}
} // namespace

void Runner::stop() {
  shouldStop_ = true;
}

Result<MethodMeta> Runner::method_meta() {
  return module_->method_meta("forward");
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
