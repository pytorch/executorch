/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama3_2/runner/runner.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::llm::Sampler;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

namespace {
static constexpr auto kTopp = 0.9f;
void printReport(const Runner::Stats& stats);
std::string statsToJsonString(const Runner::Stats& stats);
} // namespace

Runner::Runner(
    const std::vector<std::string>& models_path,
    const std::string& tokenizer_path,
    const float temperature)
    : n_bos_(1),
      n_eos_(1),
      vocab_size_(QNN_LLAMA3_2_LOGITS),
      max_seq_len_(QNN_LLAMA3_2_SEQLEN),
      tokenizer_path_(tokenizer_path),
      temperature_(temperature),
      stats_({}) {
  for (size_t i = 0; i < models_path.size(); ++i) {
    modules_.push_back(std::make_shared<Module>(
        models_path[i], Module::LoadMode::MmapUseMlockIgnoreErrors));
    ET_LOG(Info, "creating module: model_path=%s", models_path[i].c_str());
  }
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());

  tokenizer_ = example::get_tiktoken_for_llama();
  Error err = tokenizer_->load(tokenizer_path_);
  ET_CHECK_MSG(
      err == Error::Ok, "failed to load tokenizer %s", tokenizer_path_.c_str());
  eos_id_.insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
  bos_id_ = tokenizer_->bos_tok();
  eos_id_.insert(tokenizer_->eos_tok());
  io_mem_ = std::make_unique<KVCachedMemory>(modules_);
  ET_LOG(Info, "creating io_memory");
}

bool Runner::is_loaded() const {
  bool loaded = true;
  for (const std::shared_ptr<Module>& module : modules_) {
    loaded &= module->is_loaded();
  }
  return loaded && tokenizer_ && sampler_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  for (std::shared_ptr<Module>& module : modules_) {
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("forward"));
  }

  // create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  // prepare io
  auto methods_meta = get_methods_meta();
  io_mem_->prepare_io(methods_meta);
  return Error::Ok;
}

template <typename T>
int32_t Runner::logitsToToken(const Tensor& logits_tensor) {
  T* logits = logits_tensor.mutable_data_ptr<T>();

  // Since the logits are for all tokens, get the last token probabilities
  T* logits_last = logits;
  return sampler_->sample(logits_last);
}

void Runner::run_model_step(std::vector<std::vector<EValue>>& inputs) {
  for (size_t i = 0, num_modules = modules_.size(); i < num_modules; ++i) {
    Result<std::vector<EValue>> outputs_res = modules_[i]->forward(inputs[i]);
    ET_CHECK_MSG(
        outputs_res.error() == Error::Ok, "shard %zu inference failed", i);
  }
}

Error Runner::generate(
    const std::string& prompt,
    const std::string& system_prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");

  std::vector<std::vector<Tensor>> input_tensors, output_tensors;
  std::vector<std::vector<EValue>> inputs;
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    for (int i = 0; i < modules_.size(); ++i) {
      input_tensors.emplace_back(io_mem_->get_input_tensors(i));
      output_tensors.emplace_back(io_mem_->get_output_tensors(i));
      for (size_t j = 0; j < output_tensors[i].size(); ++j) {
        ET_CHECK_MSG(
            modules_[i]->set_output(output_tensors[i][j], j) == Error::Ok,
            "failed to set output tensor for module %d's %zu'th output",
            i,
            j);
      }
      inputs.emplace_back(
          std::vector<EValue>(begin(input_tensors[i]), end(input_tensors[i])));
    }
    stats_.model_load_end_ms = time_in_ms();
  }

  stats_.inference_start_ms = time_in_ms();
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  std::string post_process_prompt;

  if (!system_prompt.empty()) {
    post_process_prompt.append(
        "<|start_header_id|>system<|end_header_id|>\n\n");
    post_process_prompt.append(system_prompt);
    post_process_prompt.append("<|eot_id|>\n");
  }
  post_process_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
  post_process_prompt.append(prompt);
  post_process_prompt.append(
      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>");
  // tokenizer_->encode will add <|begin_of_text|> token for us.
  // For now, do token call back so the output format looks the same as
  // llama3 model card.
  token_callback("<|begin_of_text|>");

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(post_process_prompt, n_bos_, 0);
  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(),
      "failed to encode prompt %s",
      post_process_prompt.c_str());

  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens < max_seq_len_, "max seq length exceeded");
  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");

  int64_t pos = 0, prev_token, cur_token = prompt_tokens[0];
  KVCachedMemory::IO* ptr =
      static_cast<KVCachedMemory::IO*>(io_mem_->get_mutable_ptr());
  ptr->input_tok = static_cast<int32_t>(cur_token);
  ptr->attention_mask[max_seq_len_ - 1] = 0;

  std::vector<long long> postTime;
  while (pos < seq_len - 1) {
    // inference
    run_model_step(inputs);
    Tensor& logits_tensor = output_tensors.back()[0];

    if (pos == num_prompt_tokens) {
      stats_.first_token_ms = time_in_ms();
    } else if (pos == num_prompt_tokens - 1) {
      stats_.prompt_eval_end_ms = time_in_ms();
    }

    long sample_start_time_ms = time_in_ms();
    prev_token = cur_token;
    cur_token = logitsToToken<float>(logits_tensor);
    stats_.aggregate_sampling_time_ms += time_in_ms() - sample_start_time_ms;

    if (pos < num_prompt_tokens - 1) {
      cur_token = prompt_tokens[pos + 1];
    }
    io_mem_->update_io(cur_token, ++pos, output_tensors);
    auto piece_res = tokenizer_->decode(prev_token, cur_token);
    ET_CHECK(piece_res.ok());

    if (token_callback) {
      token_callback(piece_res.get().c_str());
    }

    if (pos >= num_prompt_tokens && eos_id_.count(cur_token) > 0) {
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = time_in_ms();
  if (pos == seq_len) {
    ET_LOG(Info, "\nSequence length (%i tokens) reached!", seq_len);
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
     << "," << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << stats.SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}
} // namespace

std::vector<Result<MethodMeta>> Runner::get_methods_meta() {
  std::vector<Result<MethodMeta>> methods_meta;
  methods_meta.reserve(modules_.size());
  for (std::shared_ptr<Module>& module : modules_) {
    methods_meta.emplace_back(module->method_meta("forward"));
  }
  return methods_meta;
}
} // namespace example
