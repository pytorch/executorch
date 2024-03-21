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
#include <executorch/extension/runner_util/managed_tensor.h>

#include <ctime>
#include <memory>
#include <sstream>

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/examples/models/llama2/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch::executor {
namespace {
static constexpr auto kTopp = 0.9f;
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
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("forward"));

  // Read out metadata: vocab_size (expected by the model), BOS, EOS, n_BOS,
  // n_EOS max_seq_len from the model
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
  use_kv_cache_ = getMetadataHelper("use_kv_cache", false);
  use_sdpa_with_kv_cache_ = getMetadataHelper("use_sdpa_with_kv_cache", false);
  append_eos_ = getMetadataHelper("append_eos_to_prompt", false);

  // Load tokenizer
  tokenizer_ = std::make_unique<Tokenizer>(vocab_size_, bos_id_, eos_id_);
  tokenizer_->load(tokenizer_path_);
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

std::vector<exec_aten::SizesType> Runner::getKVCacheShape() {
  // shape: (n_layers, args.max_batch_size, args.max_seq_len, self.n_kv_heads,
  // self.head_dim)
  std::vector<std::string> methods = {
      "get_n_layers",
      "get_max_batch_size",
      "get_max_seq_len",
      "get_n_kv_heads",
      "get_head_dim"};
  std::vector<int64_t> default_values = {12, 1, 128, 32, 128};
  std::vector<exec_aten::SizesType> result;
  for (int i = 0; i < methods.size(); ++i) {
    // convert from int64_t to int32_t
    result.push_back(getMetadataHelper<int64_t>(methods[i], default_values[i]));
  }
  return result;
}

template <typename T>
int32_t Runner::logitsToToken(
    const exec_aten::Tensor& logits_tensor,
    int64_t pos,
    T _) {
  (void)_;
  T* logits = logits_tensor.mutable_data_ptr<T>();

  // Since the logits are for all tokens, get the last token probabilities
  T* logits_last = logits;
  if (!use_kv_cache_) {
    logits_last += pos * tokenizer_->vocab_size();
  }
  return sampler_->sample(logits_last);
}

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    timers_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    timers_.model_load_end_ms = util::time_in_ms();
  }

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  timers_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  // max # of prompt tokens: len(prompt) + '\0', ?BOS, ?EOS
  int* prompt_tokens = new int[prompt.size() + 1 + n_bos_ + n_eos_];

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  tokenizer_->encode(
      prompt.c_str(),
      n_bos_,
      append_eos_ ? n_eos_ : 0,
      prompt_tokens,
      &num_prompt_tokens);

  for (int i = 0; i < num_prompt_tokens; i++) {
    ET_LOG(Info, "prompt_tokens[%d]: %d", i, prompt_tokens[i]);
  }
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  // start the main loop
  int next; // will store the next token in the sequence
  int64_t pos = num_prompt_tokens - 1; // position in the sequence
  int token = prompt_tokens[pos]; // prefill starts from 0 to num_prompt_tokens
  int logits_index = 0; // index of the logits tensor in the output
  int k_cache_index = 0;
  int v_cache_index = 0;
  std::vector<exec_aten::SizesType> kv_cache_shape = getKVCacheShape();
  std::vector<exec_aten::SizesType> input_shape = {1, 1};
  std::vector<exec_aten::SizesType> pos_shape = {1};
  std::vector<uint8_t> k_data;
  std::vector<uint8_t> v_data;
  std::vector<int64_t> token_data; // allocate space for the tokens
  std::vector<int64_t> pos_data; // allocate space for the tokens
  ScalarType dtype = static_cast<ScalarType>(
      getMetadataHelper("get_dtype", (int64_t)ScalarType::Float));

  if (use_kv_cache_) {
    // set pos to 0, refill token by token
    pos = 0;
    logits_index = 0;
    // initialize kv cache
    size_t n_bytes = 1;
    for (exec_aten::SizesType shape : kv_cache_shape) {
      n_bytes *= shape;
    }
    n_bytes *= torch::executor::elementSize(dtype);

    token_data.resize(1);
    pos_data.resize(seq_len);
  } else {
    // reserve data for tokens, notice the size is still 0.
    token_data.resize(seq_len);
  }

  // initialize tensor wrappers
  ManagedTensor pos_managed(
      pos_data.data(), pos_data.size(), pos_shape, ScalarType::Long);

  // copy prompt tokens into data
  for (int i = 0; i <= pos; ++i) {
    // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
    token_data[i] = prompt_tokens[i];
    if (i > 0) {
      printf(
          "%s",
          ET_UNWRAP(
              tokenizer_->decode(prompt_tokens[i - 1], prompt_tokens[i])));
    }
  }

  // create a 1xN int tensor with next as value
  while (pos < seq_len) {
    // ET_LOG(Info, "Generating step %d...", pos);
    // set the current token in the tensor
    std::vector<EValue> inputs;
    if (use_kv_cache_) {
      token_data[0] = token;
      input_shape[1] = 1;
      // inputs: [tokens, start_pos, k_cache, v_cache]
      inputs.emplace_back(pos_managed.get_aliasing_tensor());
    } else {
      // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
      token_data[pos] = token;
      input_shape[1] = pos + 1;
    }
    ManagedTensor token_managed(
        token_data.data(), token_data.size(), input_shape, ScalarType::Long);
    inputs.insert(inputs.begin(), token_managed.get_aliasing_tensor());
    // For kv cache, inputs: [tokens, start_pos, k_cache, v_cache]
    // Otherwise inputs: [tokens]
    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_MSG(
        outputs_res.ok(),
        "Execution of method forward failed with status 0x%" PRIx32,
        static_cast<int32_t>(outputs_res.error()));
    // ET_LOG(Info, "Model executed successfully.");

    std::vector<EValue> outputs = outputs_res.get();
    // Check the outputs.
    ET_CHECK_MSG(
        outputs.size() > 0,
        "Expecting output to have at least one evalue. Got %zu",
        outputs.size());
    if (pos == num_prompt_tokens) {
      timers_.first_token_ms = util::time_in_ms();
    } else if (pos == num_prompt_tokens - 1) {
      timers_.prompt_eval_end_ms = util::time_in_ms();
    }
    int32_t next_tok;
    exec_aten::Tensor logits_tensor = outputs.at(logits_index).toTensor();
    long sample_start_time_ms = util::time_in_ms();
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
            static_cast<int8_t>(logits_tensor.scalar_type()));
    }
    timers_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

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
    if (use_kv_cache_) {
      pos_data[0] = pos;
    }

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

    if (shouldStop_) {
      break;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens && next == eos_id_) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }

    token = next;
  }
  timers_.inference_end_ms = util::time_in_ms();
  printf("\n");

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  timers_.printReport(num_prompt_tokens, pos - num_prompt_tokens);

  delete[] prompt_tokens;
  return Error::Ok;
}

void Runner::TimeStamps::printReport(
    const int64_t& num_prompt_tokens,
    const int64_t& num_generated_tokens) {
  printf(
      "PyTorchObserver %s\n",
      toJsonString(num_prompt_tokens, num_generated_tokens).c_str());

  ET_LOG(
      Info,
      "\tPrompt Tokens: %" PRIu64 "    Generated Tokens: %" PRIu64,
      num_prompt_tokens,
      num_generated_tokens);

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t%f (seconds)",
      ((double)(model_load_end_ms - model_load_start_ms) /
       SCALING_FACTOR_UNITS_PER_SECOND));
  double inference_time_ms = (double)(inference_end_ms - inference_start_ms);
  ET_LOG(
      Info,
      "\tTotal inference time:\t\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      inference_time_ms / SCALING_FACTOR_UNITS_PER_SECOND,

      (num_generated_tokens) / (double)(inference_end_ms - inference_start_ms) *
          SCALING_FACTOR_UNITS_PER_SECOND);
  double prompt_eval_time = (double)(prompt_eval_end_ms - inference_start_ms);
  ET_LOG(
      Info,
      "\t\tPrompt evaluation:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      prompt_eval_time / SCALING_FACTOR_UNITS_PER_SECOND,
      (num_prompt_tokens) / prompt_eval_time * SCALING_FACTOR_UNITS_PER_SECOND);

  double eval_time = (double)(inference_end_ms - prompt_eval_end_ms);
  ET_LOG(
      Info,
      "\t\tGenerated %" PRIu64
      " tokens:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      num_generated_tokens,
      eval_time / SCALING_FACTOR_UNITS_PER_SECOND,
      num_generated_tokens / eval_time * SCALING_FACTOR_UNITS_PER_SECOND);

  // Time to first token is measured from the start of inference, excluding
  // model load time.
  ET_LOG(
      Info,
      "\tTime to first generated token:\t%f (seconds)",
      ((double)(first_token_ms - inference_start_ms) /
       SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tSampling time over %" PRIu64 " tokens:\t%f (seconds)",
      num_prompt_tokens + num_generated_tokens,
      (double)aggregate_sampling_time_ms / SCALING_FACTOR_UNITS_PER_SECOND);
}

const std::string Runner::TimeStamps::toJsonString(
    const int64_t& num_prompt_tokens,
    const int64_t& num_generated_tokens) {
  std::stringstream ss;
  ss << "{\"prompt_tokens\":" << num_prompt_tokens << ","
     << "\"generated_tokens\":" << num_generated_tokens << ","
     << "\"model_load_start_ms\":" << model_load_start_ms << ","
     << "\"model_load_end_ms\":" << model_load_end_ms << ","
     << "\"inference_start_ms\":" << inference_start_ms << ","
     << "\"inference_end_ms\":" << inference_end_ms << ","
     << "\"prompt_eval_end_ms\":" << prompt_eval_end_ms << ","
     << "\"first_token_ms\":" << first_token_ms << ","
     << "\"aggregate_sampling_time_ms\":" << aggregate_sampling_time_ms << ","
     << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}

void Runner::stop() {
  shouldStop_ = true;
}

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
    std::string method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
    std::string method_name,
    bool default_val);
} // namespace torch::executor
