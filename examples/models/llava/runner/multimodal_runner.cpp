/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llava/runner/multimodal_runner.h>
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
void printReport(const MultiModalRunner::Stats& stats);
std::string statsToJsonString(const MultiModalRunner::Stats& stats);
} // namespace

MultiModalRunner::MultiModalRunner(
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
      "Creating Multimodal LLM runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

bool MultiModalRunner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && sampler_;
}

Error MultiModalRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("image_encoder"));
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("token_embedding"));
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("text_model"));

  // Read out metadata: vocab_size (expected by the model), BOS, EOS, n_BOS,
  // n_EOS max_seq_len from the model
  ET_LOG(Info, "Reading metadata from model");
  const auto method_names = module_->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
  model_methods_ = method_names.get();
  n_bos_ = 1;
  n_eos_ = 1;
  max_seq_len_ = 2048;
  append_eos_ = false;

  // Load tokenizer
#if ET_USE_TIKTOKEN
  tokenizer_ = get_tiktoken_for_llama();
#else
  tokenizer_ = std::make_unique<BPETokenizer>();
#endif
  tokenizer_->load(tokenizer_path_);

  vocab_size_ = tokenizer_->vocab_size();
  bos_id_ = tokenizer_->bos_tok();
  eos_id_ = tokenizer_->eos_tok();

  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  return Error::Ok;
}

int32_t MultiModalRunner::logits_to_token(
    const exec_aten::Tensor& logits_tensor) {
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

Result<torch::executor::Tensor> MultiModalRunner::prefill_image(
    Image& image,
    int64_t start_pos) {
  ET_CHECK_MSG(!image.data.empty(), "Image cannot be empty");
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  std::vector<exec_aten::SizesType> image_shape = {
      3, image.height, image.width}; // CHW
  std::vector<exec_aten::SizesType> start_pos_shape = {1};

  ManagedTensor managed_images(
      image.data.data(), image_shape, ScalarType::Byte);
  ManagedTensor managed_start_pos(
      &start_pos, start_pos_shape, ScalarType::Long);
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  auto image_tensor = managed_images.get_aliasing_tensor();
  auto start_pos_tensor = managed_start_pos.get_aliasing_tensor();

  // image encoder input
  std::vector<EValue> image_encoder_input = {image_tensor};

  // Run image encoder
  Result<std::vector<EValue>> image_encoder_outputs =
      module_->execute("image_encoder", image_encoder_input);

  ET_CHECK_OK_OR_RETURN_ERROR(image_encoder_outputs.error());

  // inputs:[start_pos, embeds]
  std::vector<EValue> inputs;
  inputs.push_back(start_pos_tensor);
  inputs.push_back(image_encoder_outputs.get()[0]);

  // Run text model
  Result<std::vector<EValue>> outputs_res =
      module_->execute("text_model", inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing image prefill");

  // Return the logits tensor
  stats_.first_token_ms = util::time_in_ms();
  stats_.prompt_eval_end_ms = util::time_in_ms();
  return outputs_res.get()[0].toTensor();
}

Result<torch::executor::Tensor> MultiModalRunner::prefill_prompt(
    const std::string& prompt,
    int64_t start_pos,
    std::function<void(const std::string&)> token_callback) {
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
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
      "Max seq length exceeded - please increase max seq len value");

  std::vector<exec_aten::SizesType> token_shape = {1, num_prompt_tokens};
  std::vector<exec_aten::SizesType> start_pos_shape = {1};

  // initialize tensor wrappers
  ManagedTensor managed_tokens(
      prompt_tokens.data(), token_shape, ScalarType::Long);

  ManagedTensor managed_start_pos(
      &start_pos, start_pos_shape, ScalarType::Long);

  auto tokens_tensor = managed_tokens.get_aliasing_tensor();
  auto start_pos_tensor = managed_start_pos.get_aliasing_tensor();

  int32_t num_tokens = tokens_tensor.numel();

  // token embedding input
  std::vector<EValue> token_embedding_input = {tokens_tensor};

  // Run token embedding
  Result<std::vector<EValue>> token_embedding_outputs =
      module_->execute("token_embedding", token_embedding_input);
  ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_outputs.error());

  ET_LOG(
      Info,
      "Token embedding output numel(): %zu",
      token_embedding_outputs.get()[0].toTensor().numel());
  // inputs:[start_pos, tokens]
  std::vector<EValue> inputs;
  inputs.push_back(start_pos_tensor);
  inputs.push_back(token_embedding_outputs.get()[0]);

  Result<std::vector<EValue>> outputs_res =
      module_->execute("text_model", inputs);
  ET_LOG(
      Info,
      "Prefill token result numel(): %zu",
      outputs_res.get()[0].toTensor().numel());
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing LLM");
  ET_CHECK_MSG(
      outputs_res.get()[0].toTensor().size(1) == num_tokens,
      "Expected number of output tokens %d does not match returned value %zu.",
      num_tokens,
      outputs_res.get()[0].toTensor().size(1));
  if (token_callback) {
    uint64_t prev = tokens_tensor.const_data_ptr<int64_t>()[0];
    uint64_t cur;
    for (int i = 1; i < num_tokens; i++) {
      cur = tokens_tensor.const_data_ptr<int64_t>()[i];
      auto piece_res = tokenizer_->decode(prev, cur);
      ET_CHECK_OK_OR_RETURN_ERROR(piece_res.error());
      util::safe_printf(piece_res.get().c_str());
      fflush(stdout);
      prev = cur;
      token_callback(piece_res.get().c_str());
    }
    cur = logits_to_token(outputs_res.get()[0].toTensor());
    auto piece_res = tokenizer_->decode(prev, cur);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();
    util::safe_printf(piece);
    fflush(stdout);
    token_callback(piece_res.get().c_str());
  }

  // Return the logits tensor
  stats_.first_token_ms = util::time_in_ms();
  stats_.prompt_eval_end_ms = util::time_in_ms();
  return outputs_res.get()[0].toTensor();
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> MultiModalRunner::step(
    int64_t input_token,
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos,
    size_t max_seq_len) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
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
}

Error MultiModalRunner::generate(
    Image& image,
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

  // start the main loop

  // Prefill preset prompt
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  int64_t pos = 0;
  auto preset_prompt_prefill_res = prefill_prompt(kPresetPrompt, pos, nullptr);
  ET_LOG(
      Info,
      "prefill preset prompt res sizes(0): %zu, sizes(1): %zu, sizes(2): %zu",
      preset_prompt_prefill_res.get().size(0),
      preset_prompt_prefill_res.get().size(1),
      preset_prompt_prefill_res.get().size(2));

  // logits.size(1) is token length of the prompt
  pos += preset_prompt_prefill_res.get().size(1);
  ET_LOG(Info, "pos: %d", pos);

  // prefill image
  auto image_prefill_res = prefill_image(image, pos);
  ET_LOG(
      Info,
      "prefill image res sizes(0): %zu, sizes(1): %zu, sizes(2): %zu",
      image_prefill_res.get().size(0),
      image_prefill_res.get().size(1),
      image_prefill_res.get().size(2));

  // update pos to include prefilled image tokens
  pos += image_prefill_res.get().size(1);
  ET_LOG(Info, "pos: %d", pos);

  // prefill prompt
  auto prompt_prefill_res = prefill_prompt(prompt, pos, token_callback);
  ET_LOG(
      Info,
      "prefill prompt res sizes(0): %zu, sizes(1): %zu, sizes(2): %zu",
      prompt_prefill_res.get().size(0),
      prompt_prefill_res.get().size(1),
      prompt_prefill_res.get().size(2));

  // update pos to include prefilled prompt tokens
  pos += prompt_prefill_res.get().size(1);
  ET_LOG(Info, "pos: %d", pos);

  auto prefill_res_tensor = prompt_prefill_res.get();
  int64_t start_pos =
      pos; // keep a record of how many prompt tokens are prefilled
  // Generate the tokens
  int64_t prev_token;
  int64_t cur_token;
  cur_token = logits_to_token(prefill_res_tensor);
  // Prefill could be parallel or sequential.
  // Parallel:
  //  kv cache:
  //    - tokens_managed should resized to 1 as inference expects one token at
  //    a time.
  // Sequential prefill:
  //  kv cache:
  //     - tokens_managed should be resized to 1, as inference expects one
  //     token at a time.
  ManagedTensor tokens_managed(&cur_token, {1, 1}, ScalarType::Long);
  ManagedTensor start_pos_managed(&pos, {1}, ScalarType::Long);

  // Generate our tokens
  while (pos < seq_len - 1) {
    // Run the model
    Result<torch::executor::Tensor> logits_res =
        step(cur_token, tokens_managed, start_pos_managed, seq_len);

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logits_to_token(logits_tensor);
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
    if (pos >= start_pos && cur_token == eos_id_) {
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

  stats_.num_prompt_tokens = start_pos;
  stats_.num_generated_tokens = pos - start_pos;
  printReport(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

namespace {
void printReport(const MultiModalRunner::Stats& stats) {
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

std::string statsToJsonString(const MultiModalRunner::Stats& stats) {
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

void MultiModalRunner::stop() {
  shouldStop_ = true;
}

} // namespace torch::executor
