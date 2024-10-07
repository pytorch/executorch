// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <gflags/gflags.h>

// #if defined(ET_USE_THREADPOOL)
// #include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
// #include <executorch/backends/xnnpack/threadpool/threadpool.h>
// #endif

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <ctime>
#include <memory>
#include <unordered_set>
#include <utility>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

/*

The end to end flow to run this cria is as follows:
1. Build the cria model using the following command:

Get model checkpoint and tokenizer
```
manifold get
assistant_nlu/tree/users/shreyd/cria_arbitration/HTP/llama3_386M_LD12_ckpt.pt
/tmp/llama3_386M_LD12_ckpt.pt --threads 20

manifold get executorch/tree/models/llama/llama3/tokenizer.model
/tmp/tokenizer.model --threads 20
```
Generate the model given the checkpoint and params
```
buck run @mode/dev-nosan //bolt/nn/executorch/export:export_cria_model
```
It will generate a model file in the tmp directory, as described by the log

2. Build the runtime:
```
buck build @arvr/mode/android/linux/dev
//arvr/projects/bolt/bolt/nn/apps:cria_prefill_runner_app
--out /tmp
```

3. Push models and binary to device
```
adb push /tmp/cria_prefill_runner_app /vendor/bin
adb push /tmp/on_device_model.pte /data/local/tmp
adb push /tmp/tokenizer.model /data/local/tmp
```
run the binary on device
```
adb shell LD_LIBRARY_PATH=/vendor/lib64 cria_prefill_runner_app --model_path
/data/local/tmp/on_device_model.pte --tokenizer_path
/data/local/tmp/tokenizer.model
```
*/

double get_interval(
    const std::chrono::time_point<std::chrono::high_resolution_clock>& end,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return static_cast<double>(duration.count());
}

namespace torch::executor {
using Stats = ::executorch::llm::Stats;

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      float temperature = 0.8f);

  [[nodiscard]] bool is_loaded() const;
  Error load();
  Error generate(
      const std::string& prompt,
      int32_t seq_len = 128,
      const std::function<void(const std::string&)>& token_callback = {},
      const std::function<void(const Stats&)>& stats_callback = {});
  void stop();

 private:
  // metadata
  template <typename T>
  T getMetadataHelper(const std::string& method_name, T default_val);
  int32_t logitsToToken(const exec_aten::Tensor& logits_tensor);
  Result<torch::executor::Tensor> prefill(
      const std::vector<uint64_t>& tokens,
      executorch::extension::TensorPtr& managed_tokens,
      executorch::extension::TensorPtr& managed_start_pos,
      const std::function<void(const std::string&)>& token_callback);
  Result<torch::executor::Tensor> run_model_step(
      int64_t input_token,
      executorch::extension::TensorPtr& tokens,
      executorch::extension::TensorPtr& start_pos,
      size_t max_seq_len);
  // metadata
  int32_t vocab_size_{};
  int32_t bos_id_{};
  int32_t eos_id_{};
  int32_t n_bos_{};
  int32_t n_eos_{};
  int32_t max_seq_len_{};
  bool append_eos_{};
  std::unordered_set<std::string> model_methods_;
  std::string model_path_;
  std::unique_ptr<Module> module_;
  std::string tokenizer_path_;
  float temperature_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
  bool shouldStop_{false};
  Stats stats_;
  bool enable_parallel_prefill_{};
};

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
  // max_seq_len_ = getMetadataHelper<int64_t>("get_max_seq_len", 33);
  max_seq_len_ = 17;
  append_eos_ = getMetadataHelper("append_eos_to_prompt", false);
  enable_parallel_prefill_ = getMetadataHelper("enable_dynamic_shape", false);

  tokenizer_ = example::get_tiktoken_for_llama();
  tokenizer_->load(tokenizer_path_);

  vocab_size_ =
      getMetadataHelper<int64_t>("get_vocab_size", tokenizer_->vocab_size());
  bos_id_ = getMetadataHelper<int64_t>("get_bos_id", tokenizer_->bos_tok());
  eos_id_ = getMetadataHelper<int64_t>("get_eos_id", tokenizer_->eos_tok());

  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      ::executorch::llm::kTopp,
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
      if (!outs.empty()) {
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
      auto* logits = logits_tensor.mutable_data_ptr<float>();
      float* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last);
    }
    case ScalarType::Half: {
      auto* logits = logits_tensor.mutable_data_ptr<exec_aten::Half>();
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
    executorch::extension::TensorPtr& managed_tokens,
    executorch::extension::TensorPtr& /*managed_start_pos*/,

    const std::function<void(const std::string&)>& token_callback) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_tokens = tokens.size();
  ET_LOG(Info, "Prefilling %d tokens", num_tokens);

  ET_CHECK_OK_OR_RETURN_ERROR(executorch::extension::resize_tensor_ptr(
      managed_tokens, {1, num_tokens}));
  auto* tokens_ptr = managed_tokens->mutable_data_ptr<int64_t>();
  for (int i = 0; i < num_tokens; i++) {
    // The following assumes batch size = 1
    tokens_ptr[i] = tokens[i];
  }
  std::vector<EValue> inputs;

  // inputs:[tokens, start_pos]
  inputs.emplace_back(managed_tokens);
  // inputs.push_back(start_pos);

  auto before_exec = std::chrono::high_resolution_clock::now();
  Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
  auto after_exec = std::chrono::high_resolution_clock::now();
  ET_LOG(Info, "execute took %f ms", get_interval(after_exec, before_exec));

  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing LLM");
  ET_CHECK_MSG(
      outputs_res.get()[0].toTensor().size(1) == num_tokens,
      "Expected number of output tokens %d does not match returned value %zu.",
      num_tokens,
      outputs_res.get()[0].toTensor().size(1));

  // start_pos.mutable_data_ptr<int64_t>()[0] = num_tokens;

  uint64_t prev = tokens[0];
  uint64_t cur = 0;
  for (int i = 1; i < num_tokens; i++) {
    cur = tokens[i];
    auto piece_res = tokenizer_->decode(prev, cur);
    ET_CHECK_OK_OR_RETURN_ERROR(piece_res.error());
    util::safe_printf(piece_res.get().c_str());
    fflush(stdout);
    prev = cur;
    if (token_callback) {
      token_callback(piece_res.get());
    }
  }
  cur = logitsToToken(outputs_res.get()[0].toTensor());
  auto piece_res = tokenizer_->decode(prev, cur);
  ET_CHECK(piece_res.ok());
  const char* piece = piece_res.get().c_str();
  util::safe_printf(piece);
  fflush(stdout);
  if (token_callback) {
    token_callback(piece_res.get());
  }

  // Return the logits tensor
  stats_.first_token_ms = util::time_in_ms();
  stats_.prompt_eval_end_ms = util::time_in_ms();
  return outputs_res.get()[0].toTensor();
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> Runner::run_model_step(
    int64_t input_token,
    executorch::extension::TensorPtr& tokens,
    executorch::extension::TensorPtr& start_pos,
    size_t max_seq_len) {
  std::vector<EValue> inputs;
  (void)start_pos; // unused

  // When not using kv-cache our input is the entire history of tokens we have
  // seen, so resize input to be 1 larger and append the new token to the end.
  // TODO does this work in ATen mode?
  tokens->mutable_data_ptr<int64_t>()[tokens->size(1) - 1] = input_token;

  // inputs:[tokens]
  inputs.emplace_back(tokens);

  auto outputs_res = module_->forward(inputs);

  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get().size() == 1,
      "More then one output returned from executing LLM.");
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing LLM");

  if (tokens->size(1) < max_seq_len) {
    // Resize the tokens tensor to be 1 larger for next step.
    // Note that this relies on the fact that underlying memory is the same
    // such that previous tokens stored there will still exist.
    // Not a good thing to rely upon.
    ET_CHECK_OK_OR_RETURN_ERROR(executorch::extension::resize_tensor_ptr(
        tokens, {1, static_cast<int>(tokens->size(1) + 1)}));
  }

  // Return the logits tensor
  return outputs_res.get()[0].toTensor();
}

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    const std::function<void(const std::string&)>& token_callback,
    const std::function<void(const Stats&)>& stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  // auto generate_start = std::chrono::high_resolution_clock::now();
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = util::time_in_ms();
    ET_LOG(Info, "[Time consuming during load() function] init took %ld ms", stats_.model_load_end_ms-stats_.model_load_start_ms);
  }

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  // auto encode_start = std::chrono::high_resolution_clock::now();

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

  // auto encode_finish = std::chrono::high_resolution_clock::now();
  // ET_LOG(Info, "encoder took %f ms", get_interval(encode_finish,
  // encode_start));

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  ET_LOG(Info, "Prompt tokens: %zu", prompt_tokens.size());
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py num_prompt_tokens: %d, max_seq_len: %d",
      num_prompt_tokens,
      max_seq_len_);

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

  // initialize tensor wrappers
  auto tokens = executorch::extension::from_blob(
      token_data.data(), token_shape, ScalarType::Long);
  // Create with the max shape to approapriately set the capacity of this
  // tensor, then resize back to 1 for first input.
  ET_CHECK_OK_OR_RETURN_ERROR(
      executorch::extension::resize_tensor_ptr(tokens, {1, 1}));

  auto start_pos = executorch::extension::from_blob(
      start_pos_data.data(), start_pos_shape, ScalarType::Long);

  int64_t cur_token = prompt_tokens[0];

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res = prefill(prompt_tokens, tokens, start_pos, token_callback);

  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  exec_aten::Tensor& prefill_res_tensor = prefill_res.get();
  cur_token = logitsToToken(prefill_res_tensor);
  ET_LOG(Info, "Prefill result: %ld", cur_token);

  ET_CHECK_OK_OR_RETURN_ERROR(executorch::extension::resize_tensor_ptr(
      tokens, {1, num_prompt_tokens + 1}));
  // tokens_managed.resize({1, num_prompt_tokens + 1});
  pos = num_prompt_tokens;

  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = pos - num_prompt_tokens;
  ::executorch::llm::print_report(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

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

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(
    prompt,
    "How does artificial intelligence redefine the role of human creativity in the next decade?",
    "Prompt.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    17,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();
  ET_LOG(Info, "Prompt: %s", prompt);

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  // [[maybe_unused]] int32_t cpu_threads = FLAGS_cpu_threads;

  // #if defined(ET_USE_THREADPOOL)
  //   uint32_t num_performant_cores = cpu_threads == -1
  //       ? torch::executorch::cpuinfo::get_num_performant_cores()
  //       : static_cast<uint32_t>(cpu_threads);
  //   ET_LOG(Info, "Resetting threadpool with num threads = %d",
  //   num_performant_cores); if (num_performant_cores > 0) {
  //     torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(num_performant_cores);
  //   }
  // #endif
  // create llama runner
  ::torch::executor::Runner runner(model_path, tokenizer_path, temperature);

  // generate
  runner.generate(prompt, seq_len);

  return 0;
}
