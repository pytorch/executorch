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
#if ET_USE_TIKTOKEN
#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#else /* BPE */
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#endif /* ET_USE_TIKTOKEN*/
#include <executorch/extension/evalue_util/print_evalue.h>
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
void printReport(const Runner::Stats& stats);
std::string statsToJsonString(const Runner::Stats& stats);
} // namespace

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
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
  n_bos_ = getMetadataHelper<int64_t>("get_n_bos", 1);
  n_eos_ = getMetadataHelper<int64_t>("get_n_eos", 1);
  max_seq_len_ = getMetadataHelper<int64_t>("get_max_seq_len", 128);
  use_kv_cache_ = getMetadataHelper("use_kv_cache", true);
  use_sdpa_with_kv_cache_ = getMetadataHelper("use_sdpa_with_kv_cache", false);
  append_eos_ = getMetadataHelper("append_eos_to_prompt", false);
  enable_parallel_prefill_ = getMetadataHelper("enable_dynamic_shape", false);

  // Load tokenizer
#if ET_USE_TIKTOKEN
  tokenizer_ = get_tiktoken_for_llama();
#else
  tokenizer_ = std::make_unique<BPETokenizer>();
#endif
  tokenizer_->load(tokenizer_path_);

  vocab_size_ =
      getMetadataHelper<int64_t>("get_vocab_size", tokenizer_->vocab_size());
  bos_id_ = getMetadataHelper<int64_t>("get_bos_id", tokenizer_->bos_tok());
  eos_id_ = getMetadataHelper<int64_t>("get_eos_id", tokenizer_->eos_tok());

  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  return Error::Ok;
}

template <typename T>
T Runner::getMetadataHelper(const std::string& method_name, T default_val) {
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

int32_t Runner::logitsToToken(const exec_aten::Tensor& logits_tensor) {
  ET_CHECK_MSG(logits_tensor.dim() == 3, "Logits tensor must be 3D");
  auto num_tokens = logits_tensor.size(1);

  switch (logits_tensor.scalar_type()) {
    case ScalarType::Float: {
      float* logits = logits_tensor.mutable_data_ptr<float>();
      float* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last);
    }
    case ScalarType::Half: {
      exec_aten::Half* logits =
          logits_tensor.mutable_data_ptr<exec_aten::Half>();
      exec_aten::Half* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last);
    }
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported dtype output %hhd",
          static_cast<int8_t>(logits_tensor.scalar_type()));
  }
}

Result<torch::executor::Tensor> Runner::prefill(
    const std::vector<uint64_t>& tokens,
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos,
    std::function<void(const std::string&)> token_callback) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_tokens = tokens.size();
  if (enable_parallel_prefill_) {
    managed_tokens.resize({1, num_tokens});
    int64_t* tokens_ptr =
        managed_tokens.get_aliasing_tensor().mutable_data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; i++) {
      // The following assumes batch size = 1
      tokens_ptr[i] = tokens[i];
    }
    std::vector<EValue> inputs;
    auto tokens_tensor = managed_tokens.get_aliasing_tensor();
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    // inputs:[tokens, start_pos]
    inputs.push_back(tokens_tensor);
    inputs.push_back(start_pos);

    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");
    ET_CHECK_MSG(
        outputs_res.get()[0].toTensor().size(1) == num_tokens,
        "Expected number of output tokens %d does not match returned value %zu.",
        num_tokens,
        outputs_res.get()[0].toTensor().size(1));

    start_pos.mutable_data_ptr<int64_t>()[0] = num_tokens;

    uint64_t prev = tokens[0];
    uint64_t cur;
    for (int i = 1; i < num_tokens; i++) {
      cur = tokens[i];
      auto piece_res = tokenizer_->decode(prev, cur);
      ET_CHECK_OK_OR_RETURN_ERROR(piece_res.error());
      util::safe_printf(piece_res.get().c_str());
      fflush(stdout);
      prev = cur;
      if (token_callback) {
        token_callback(piece_res.get().c_str());
      }
    }
    cur = logitsToToken(outputs_res.get()[0].toTensor());
    auto piece_res = tokenizer_->decode(prev, cur);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();
    util::safe_printf(piece);
    fflush(stdout);
    if (token_callback) {
      token_callback(piece_res.get().c_str());
    }

    // Return the logits tensor
    stats_.first_token_ms = util::time_in_ms();
    stats_.prompt_eval_end_ms = util::time_in_ms();
    return outputs_res.get()[0].toTensor();
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    int64_t cur_token = tokens[0];
    int64_t prev_token;
    // This is a hack to enable returning a logits tensor from prefill
    auto logits_tensor = managed_tokens.get_aliasing_tensor();
    while (pos < num_tokens) {
      // Run the model
      Result<torch::executor::Tensor> logits_res = run_model_step(
          cur_token, managed_tokens, managed_start_pos, num_tokens);

      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      logits_tensor = logits_res.get();
      // Hack to enable returning a logits tensor from prefill

      prev_token = cur_token;

      long sample_start_time_ms = util::time_in_ms();
      cur_token = logitsToToken(logits_tensor);
      stats_.aggregate_sampling_time_ms +=
          util::time_in_ms() - sample_start_time_ms;

      // advance the state machine
      if (pos < num_tokens - 1) {
        // prefill, force the next token to be the next prompt token
        cur_token = tokens[pos + 1];
      }
      pos++;

      // print the token as string, decode it with the Tokenizer object
      auto piece_res = tokenizer_->decode(prev_token, cur_token);
      ET_CHECK(piece_res.ok());
      const char* piece = piece_res.get().c_str();
      util::safe_printf(piece);
      fflush(stdout);
      if (token_callback) {
        token_callback(piece_res.get().c_str());
      }
    }
    auto start_pos = managed_start_pos.get_aliasing_tensor();
    start_pos.mutable_data_ptr<int64_t>()[0] = num_tokens;
    stats_.first_token_ms = util::time_in_ms();
    stats_.prompt_eval_end_ms = util::time_in_ms();
    return logits_tensor;
  }
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> Runner::run_model_step(
    int64_t input_token,
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos,
    size_t max_seq_len) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  if (use_kv_cache_) {
    auto tokens = managed_tokens.get_aliasing_tensor();
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    // When using kv-cache our input is always 1 token, so just update to the
    // latest.
    tokens.mutable_data_ptr<int64_t>()[0] = input_token;

    Result<std::vector<EValue>> outputs_res =
        module_->forward({tokens, start_pos});
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Bump start_pos by 1
    start_pos.mutable_data_ptr<int64_t>()[0]++;

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    std::vector<EValue> inputs;
    auto tokens = managed_tokens.get_aliasing_tensor();
    (void)managed_start_pos; // unused

    // When not using kv-cache our input is the entire history of tokens we have
    // seen, so resize input to be 1 larger and append the new token to the end.
    // TODO does this work in ATen mode?
    tokens.mutable_data_ptr<int64_t>()[tokens.size(1) - 1] = input_token;

    // inputs:[tokens]
    inputs.push_back(tokens);

    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    if (tokens.size(1) < max_seq_len) {
      // Resize the tokens tensor to be 1 larger for next step.
      // Note that this relies on the fact that underlying memory is the same
      // such that previous tokens stored there will still exist.
      // Not a good thing to rely upon.
      managed_tokens.resize({1, static_cast<int>(tokens.size(1) + 1)});
    }

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  }
}

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = util::time_in_ms();
  }

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  // start the main loop
  int64_t pos = 0; // position in the sequence

  std::vector<int64_t> token_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> token_shape = {1, seq_len};

  std::vector<int64_t> start_pos_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> start_pos_shape = {1};

  token_data.resize(seq_len);
  if (use_kv_cache_) {
    // hard code these to size 1 as kv cache is locked to static size right now.
    start_pos_data.resize(1);
    start_pos_data.push_back(0);
  }

  // initialize tensor wrappers
  ManagedTensor tokens_managed(
      token_data.data(), token_shape, ScalarType::Long);
  // Create with the max shape to approapriately set the capacity of this
  // tensor, then resize back to 1 for first input.
  tokens_managed.resize({1, 1});

  ManagedTensor start_pos_managed(
      start_pos_data.data(), start_pos_shape, ScalarType::Long);

  int64_t prev_token;
  int64_t cur_token = prompt_tokens[0];

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res =
      prefill(prompt_tokens, tokens_managed, start_pos_managed, token_callback);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  exec_aten::Tensor& prefill_res_tensor = prefill_res.get();
  cur_token = logitsToToken(prefill_res_tensor);
  if (use_kv_cache_) {
    // Prefill could be parallel or sequential.
    // Parallel:
    //  kv cache:
    //    - tokens_managed should resized to 1 as inference expects one token at
    //    a time.
    //  no kv cache:
    //    - tokens_managed should be resized to prompt length + 1, as inference
    //    expects all tokens at once.
    // Sequential prefill:
    //  kv cache:
    //     - tokens_managed should be resized to 1, as inference expects one
    //     token at a time.
    //  no kv cache:
    //     - tokens_managed should be resized to prompt length + 1, as inference
    //     expects all tokens at once.
    tokens_managed.resize({1, 1});
  } else {
    tokens_managed.resize({1, num_prompt_tokens + 1});
  }
  pos = num_prompt_tokens;

  // Generate our tokens
  while (pos < seq_len - 1) {
    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(cur_token, tokens_managed, start_pos_managed, seq_len);

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logitsToToken(logits_tensor);
    stats_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

    pos++;

    // print the token as string, decode it with the Tokenizer object
    auto piece_res = tokenizer_->decode(prev_token, cur_token);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();

    // same as printf("%s", piece), but skips "unsafe" bytes
    util::safe_printf(piece);
    fflush(stdout);

    if (token_callback) {
      token_callback(piece);
    }

    if (shouldStop_) {
      break;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens && cur_token == eos_id_) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

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
     << "," << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << stats.SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}
} // namespace

void Runner::stop() {
  shouldStop_ = true;
}

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
    const std::string& method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
    const std::string& method_name,
    bool default_val);
} // namespace torch::executor
