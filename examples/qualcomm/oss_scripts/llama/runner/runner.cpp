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
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <ctime>
#include <fstream>
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
    const float logits_scale,
    const int32_t logits_offset,
    const float temperature,
    const int eval_mode,
    const std::string& kv_updator)
    : n_bos_(1),
      n_eos_(1),
      tokenizer_path_(tokenizer_path),
      logits_scale_(logits_scale),
      logits_offset_(logits_offset),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      kv_updator_(kv_updator) {
  for (size_t i = 0; i < models_path.size(); ++i) {
    modules_.push_back(std::make_shared<Module>(
        models_path[i], Module::LoadMode::MmapUseMlockIgnoreErrors));
    ET_LOG(Info, "creating module: model_path=%s", models_path[i].c_str());
  }
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
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

  switch (eval_mode_) {
    case EvalMode::kPrefill:
      prefill_forward_name_ = "forward";
      method_names_.emplace_back(prefill_forward_name_);
      break;
    case EvalMode::kKVCached:
      kv_forward_name_ = "forward";
      method_names_.emplace_back(kv_forward_name_);
      break;
    case EvalMode::kHybrid:
      prefill_forward_name_ = "prefill_forward";
      kv_forward_name_ = "kv_forward";
      method_names_.emplace_back(prefill_forward_name_);
      method_names_.emplace_back(kv_forward_name_);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama version");
      break;
  }

  for (std::shared_ptr<Module>& module : modules_) {
    if (!prefill_forward_name_.empty()) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(prefill_forward_name_));
    }
    if (!kv_forward_name_.empty()) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(kv_forward_name_));
    }
  }

  if (!prefill_forward_name_.empty()) {
    // Use input tokens length to retrieve prefill cache len
    // Cache len equals to prefill model seq_len - 1
    prefill_cache_len_ = get_methods_meta(prefill_forward_name_)[0]
                             ->input_tensor_meta(0)
                             ->sizes()[1];
  }
  if (!kv_forward_name_.empty()) {
    // Use k cache length to retirieve kv cache len
    // Cache len equals to kv model seq_len - 1
    kv_cache_len_ =
        get_methods_meta(kv_forward_name_)[0]->input_tensor_meta(3)->sizes()[2];
  }

  // retrieve any method meta, can be either prefill or kv
  // Try avoid getMetadataHelper as it is time consuming.
  auto method_meta = get_methods_meta(method_names_[0])[0].get();
  int64_t num_layers = getMetadataHelper<int64_t>("get_n_layers", -1);
  int64_t head_dim = method_meta.output_tensor_meta(1)->sizes()[1]; // k_cache
  int64_t num_heads = (method_meta.num_outputs() - 1) / (num_layers * 2);
  vocab_size_ = method_meta.output_tensor_meta(0)->sizes()[2]; // logit_tensor
  use_int64_token_ = method_meta.input_tensor_meta(0)->scalar_type() ==
      executorch::aten::ScalarType::Long;
  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");

  if (kv_updator_ == "SmartMask") {
    io_mgr_ = std::make_unique<SmartMaskIoMgr>(
        modules_,
        prefill_cache_len_,
        kv_cache_len_,
        vocab_size_,
        num_layers,
        head_dim,
        num_heads,
        eval_mode_,
        prefill_forward_name_,
        kv_forward_name_,
        use_int64_token_);
  } else if (kv_updator_ == "ShiftPointer") {
    io_mgr_ = std::make_unique<ShiftPointerIoMgr>(
        modules_,
        prefill_cache_len_,
        kv_cache_len_,
        vocab_size_,
        num_layers,
        head_dim,
        num_heads,
        eval_mode_,
        prefill_forward_name_,
        kv_forward_name_,
        use_int64_token_);
  } else {
    ET_LOG(Error, "Using an unknown updator %s", kv_updator_.c_str());
  }
  ET_LOG(Info, "creating io_memory");

  // prepare io
  io_mgr_->init_io();
  switch (eval_mode_) {
    case EvalMode::kPrefill:
      io_mgr_->prepare_prefill_io(get_methods_meta(prefill_forward_name_));
      break;
    case EvalMode::kKVCached:
      io_mgr_->prepare_kv_io(get_methods_meta(kv_forward_name_));
      break;
    case EvalMode::kHybrid:
      io_mgr_->prepare_prefill_io(get_methods_meta(prefill_forward_name_));
      io_mgr_->prepare_kv_io(get_methods_meta(kv_forward_name_));
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "unsupported mode");
      break;
  }

  // llama3 tokenizer
  tokenizer_ = example::get_tiktoken_for_llama();
  Error err = tokenizer_->load(tokenizer_path_);
  if (err == Error::InvalidArgument) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken artifact, trying BPE tokenizer",
        tokenizer_path_.c_str());
    tokenizer_.reset();
    // llama2 tokenizer
    tokenizer_ = std::make_unique<executorch::extension::llm::BPETokenizer>();
    err = tokenizer_->load(tokenizer_path_);
    llama_version_ = LlamaVersion::kLlama2;
    ET_CHECK_MSG(
        err == Error::Ok,
        "failed to load tokenizer %s",
        tokenizer_path_.c_str());
  } else {
    eos_id_.insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
    llama_version_ = LlamaVersion::kLlama3;
  }
  bos_id_ = tokenizer_->bos_tok();
  eos_id_.insert(tokenizer_->eos_tok());

  // create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  return Error::Ok;
}

template <typename T>
T Runner::getMetadataHelper(std::string method_name, T default_val) {
  T res = default_val;
  if (modules_[0]->method_names()->count(method_name)) {
    Result<std::vector<EValue>> outputs = modules_[0]->execute(method_name);
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
  return res;
}

int32_t Runner::logitsToToken(const Tensor& logits_tensor, int64_t pos) {
  static std::vector<float> logits_f(vocab_size_);
  const uint16_t* logits = logits_tensor.data_ptr<uint16_t>();
  // Since the logits are for all tokens, get the last token probabilities
  auto* logits_last = logits;

  // offset to the meaningful logit we want.
  if (logits_tensor.sizes().data()[1] > 1) {
    logits_last += pos * vocab_size_;
  }

  // dequantize
  for (int i = 0; i < vocab_size_; i++) {
    logits_f[i] = (logits_last[i] - logits_offset_) * logits_scale_;
  }
  return sampler_->sample(logits_f.data());
}

void Runner::run_model_step(
    const std::string& method_name,
    std::vector<std::vector<EValue>>& inputs) {
  for (size_t i = 0, num_modules = modules_.size(); i < num_modules; ++i) {
    Result<std::vector<EValue>> outputs_res =
        modules_[i]->execute(method_name, inputs[i]);
    ET_CHECK_MSG(
        outputs_res.error() == Error::Ok, "shard %zu inference failed", i);
  }
}

Error Runner::generate(
    int32_t seq_len,
    const std::string& prompt,
    const std::string& system_prompt,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  std::unordered_map<std::string, std::vector<std::vector<Tensor>>>
      input_tensors, output_tensors;
  std::unordered_map<std::string, std::vector<std::vector<EValue>>> inputs;
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    for (auto method_name : method_names_) {
      for (int i = 0; i < modules_.size(); ++i) {
        input_tensors[method_name].emplace_back(
            io_mgr_->get_input_tensors(i, method_name));
        output_tensors[method_name].emplace_back(
            io_mgr_->get_output_tensors(i, method_name));
        for (size_t j = 0; j < output_tensors[method_name][i].size(); ++j) {
          ET_CHECK_MSG(
              modules_[i]->set_output(
                  method_name, output_tensors[method_name][i][j], j) ==
                  Error::Ok,
              "failed to set output tensor for module %d's %zu'th output",
              i,
              j);
        }
        inputs[method_name].emplace_back(std::vector<EValue>(
            begin(input_tensors[method_name][i]),
            end(input_tensors[method_name][i])));
      }
    }
  }
  stats_.model_load_end_ms = time_in_ms();
  stats_.inference_start_ms = time_in_ms();

  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");

  switch (llama_version_) {
    case LlamaVersion::kLlama2:
      prompt_.append(prompt);
      break;
    case LlamaVersion::kLlama3:
      if (!system_prompt.empty()) {
        prompt_.append("<|start_header_id|>system<|end_header_id|>\n\n");
        prompt_.append(system_prompt);
        prompt_.append("<|eot_id|>");
      }
      prompt_.append("<|start_header_id|>user<|end_header_id|>\n\n");
      prompt_.append(prompt);
      prompt_.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      if (token_callback) {
        token_callback("<|begin_of_text|>");
      }
      break;
    default:
      ET_CHECK_MSG(false, "unsupported llama version");
      break;
  }

  int max_seq_len = std::max(prefill_cache_len_, kv_cache_len_) + 1;
  seq_len = (seq_len > 0 && seq_len <= max_seq_len) ? seq_len : max_seq_len;
  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt_, n_bos_, 0);
  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "failed to encode prompt %s", prompt_.c_str());

  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens < max_seq_len, "max seq length exceeded");
  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");
  if (eval_mode_ == EvalMode::kHybrid) {
    int prefill_seq_len = get_methods_meta(prefill_forward_name_)[0]
                              ->input_tensor_meta(0)
                              ->sizes()[1] +
        1;
    ET_CHECK_MSG(
        num_prompt_tokens < prefill_seq_len,
        "For hybrid mode, please ensure prompt length(%d) is less than prefill's seq_len(%d)",
        num_prompt_tokens,
        prefill_seq_len);
  }

  int64_t pos = 0, prev_token, cur_token = prompt_tokens[0];
  if (token_callback) {
    token_callback(prompt_);
  }
  auto prefill_execute = [&](const std::string& method_name) {
    io_mgr_->fill_prefill_toks(prompt_tokens);

    pos = num_prompt_tokens - 1;
    cur_token = prompt_tokens[pos];
    while (pos < seq_len - 1) {
      // inference
      run_model_step(method_name, inputs[method_name]);
      Tensor& logits_tensor = output_tensors[method_name].back()[0];
      prev_token = cur_token;
      long sample_start_time_ms = time_in_ms();
      cur_token = logitsToToken(logits_tensor, pos);
      stats_.aggregate_sampling_time_ms += time_in_ms() - sample_start_time_ms;

      io_mgr_->update_prefill_io(cur_token, ++pos, output_tensors[method_name]);
      auto piece_res = tokenizer_->decode(prev_token, cur_token);
      ET_CHECK(piece_res.ok());
      if (token_callback) {
        token_callback(piece_res.get().c_str());
      }

      if (pos == num_prompt_tokens) {
        stats_.first_token_ms = time_in_ms();
        stats_.prompt_eval_end_ms = time_in_ms();
      }

      if (pos >= num_prompt_tokens && eos_id_.count(cur_token) > 0) {
        ET_LOG(Info, "\nReached to the end of generation");
        break;
      }
      // prefill model inferences once for prompt in the hybrid mode
      if (eval_mode_ == EvalMode::kHybrid) {
        break;
      }
    }
  };

  auto kv_execute = [&](const std::string& method_name) {
    io_mgr_->fill_kv_tok_mask(pos, cur_token);
    while (pos < seq_len - 1) {
      // inference
      run_model_step(method_name, inputs[method_name]);
      Tensor& logits_tensor = output_tensors[method_name].back()[0];

      // hybrid mode will check these stats_ at prefill(prefill)
      if (eval_mode_ == EvalMode::kKVCached) {
        if (pos == num_prompt_tokens) {
          stats_.first_token_ms = time_in_ms();
        } else if (pos == num_prompt_tokens - 1) {
          stats_.prompt_eval_end_ms = time_in_ms();
        }
      }
      prev_token = cur_token;
      long sample_start_time_ms = time_in_ms();
      cur_token = logitsToToken(logits_tensor, pos);
      stats_.aggregate_sampling_time_ms += time_in_ms() - sample_start_time_ms;

      if (pos < num_prompt_tokens - 1) {
        cur_token = prompt_tokens[pos + 1];
      }
      io_mgr_->update_kv_io(cur_token, ++pos, output_tensors[method_name]);
      auto piece_res = tokenizer_->decode(prev_token, cur_token);
      ET_CHECK(piece_res.ok());

      if (token_callback && pos >= num_prompt_tokens) {
        token_callback(piece_res.get().c_str());
      }

      if (pos >= num_prompt_tokens && eos_id_.count(cur_token) > 0) {
        ET_LOG(Info, "\nReached to the end of generation");
        break;
      }
    }
  };

  switch (eval_mode_) {
    case EvalMode::kPrefill:
      prefill_execute(prefill_forward_name_);
      break;
    case EvalMode::kKVCached:
      kv_execute(kv_forward_name_);
      break;
    case EvalMode::kHybrid:
      prefill_execute(prefill_forward_name_);
      io_mgr_->update_prefill_to_kv_io(
          cur_token, pos, output_tensors[kv_forward_name_]);
      kv_execute(kv_forward_name_);
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported eval mode");
      break;
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
      stats.num_generated_tokens,
      (double)stats.aggregate_sampling_time_ms /
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile("outputs/inference_speed.txt");
  if (outfile.is_open()) {
    double num_tok = (stats.num_generated_tokens) /
        (double)(stats.inference_end_ms - stats.inference_start_ms) *
        stats.SCALING_FACTOR_UNITS_PER_SECOND;
    outfile << num_tok;
    outfile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the inference speed file");
  }
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

std::vector<Result<MethodMeta>> Runner::get_methods_meta(
    std::string& method_name) {
  std::vector<Result<MethodMeta>> methods_meta;
  methods_meta.reserve(modules_.size());
  for (std::shared_ptr<Module>& module : modules_) {
    methods_meta.emplace_back(module->method_meta(method_name));
  }
  return methods_meta;
}
} // namespace example
