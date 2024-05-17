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
    const std::vector<std::string>& model_path_list,
    const std::string& tokenizer_path,
    const float temperature)
    : tokenizer_path_(tokenizer_path),
      temperature_(temperature) {
  for(auto& model_path : model_path_list){
    modules_.emplace_back(std::make_unique<Module>(
          model_path,
          Module::MlockConfig::UseMlockIgnoreErrors));
    ET_LOG(
        Info,
        "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
        model_path.c_str(),
        tokenizer_path.c_str());
  }
}

bool Runner::is_loaded() const {
  bool loaded = true;
  for(auto& module : modules_){
    loaded &= module->is_loaded();
  }
  return loaded && tokenizer_ && sampler_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_.model_load_start_ms = util::time_in_ms();
  for(auto& module : modules_){
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("forward"));


    // Read out metadata from the model
    ET_LOG(Info, "Reading metadata from model");
    const auto method_names = module->method_names();
    ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
    model_methods_ = method_names.get();
    vocab_size_ = getMetadataHelper<int64_t>(module.get(), "get_vocab_size", 32000);
    bos_id_ = getMetadataHelper<int64_t>(module.get(), "get_bos_id", 1);
    eos_id_ = getMetadataHelper<int64_t>(module.get(), "get_eos_id", 2);
    n_bos_ = getMetadataHelper<int64_t>(module.get(), "get_n_bos", 1);
    n_eos_ = getMetadataHelper<int64_t>(module.get(), "get_n_eos", 1);
    max_seq_len_ = getMetadataHelper<int64_t>(module.get(), "get_max_seq_len", 128);
    head_dim_ = getMetadataHelper<int64_t>(module.get(), "get_head_dim", 32);
  }
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
T Runner::getMetadataHelper(Module* module, std::string method_name, T default_val) {
  T res = default_val;
  if (model_methods_.count(method_name)) {
    Result<std::vector<EValue>> outputs = module->execute(method_name);
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
    Tensor& atten_mask,
    Tensor& freqs_cos,
    Tensor& freqs_sin,
    std::vector<std::vector<Tensor>>& kv_tensors,
    std::vector<std::vector<Tensor>>& kv_outputs) {
  token.mutable_data_ptr<int32_t>()[0] = input_token;
  // embedding
  std::vector<EValue> inputs = {token};
  Result<std::vector<EValue>> outputs_res = modules_[0]->forward(inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  EValue hidden_states = outputs_res.get()[0];

  // llama block
  std::vector<Result<std::vector<EValue>>> llama_block_results;
  for(int i = 1; i < modules_.size() - 1; ++i){
    inputs = {hidden_states, freqs_cos, freqs_sin, atten_mask};
    inputs.insert(inputs.end(), kv_tensors[i-1].begin(), kv_tensors[i-1].end());
    Result<std::vector<EValue>> llama_block_outputs_res = modules_[i]->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(llama_block_outputs_res.error());
    hidden_states = llama_block_outputs_res.get()[0];
  }

  // TODO: need to handle batch size != 1
  // update k_cache
  size_t v_offset = kv_outputs[0][0].nbytes();
  size_t el_size = kv_outputs[0][0].element_size();
  size_t k_input_step = (max_seq_len_-1) * el_size;
  for (int i = 1; i < modules_.size() - 1; ++i) {
    int k_tensors_end = kv_tensors[i].size() / 2;
    //update k caches
    for (int j = 0, index = i-1; j < k_tensors_end; ++j) {
      char *input_addr = static_cast<char*>(kv_tensors[index][j].mutable_data_ptr());
      char *output_addr = static_cast<char*>(kv_outputs[index][j].mutable_data_ptr());

      // fill the output k values back
      #pragma omp parallel for
      for (int src = 0, dst = k_input_step; src < kv_outputs[index][j].nbytes(); src+=el_size, dst+=k_input_step) {
        memcpy(input_addr+dst, output_addr+src, el_size);
      }

      // inputs
      ET_CHECK_MSG(
          internal::set_tensor_data(kv_tensors[index][j], input_addr + kv_tensors[index][j].element_size(), kv_tensors[index][j].nbytes()) == Error::Ok,
          "Failed to set input tensor when updating kv_cache");
    }
    // update v caches
    for (int j = k_tensors_end, index = i-1; j < kv_tensors[index].size(); ++j) {
      // inputs
      char *input_addr = static_cast<char*>(kv_tensors[index][j].mutable_data_ptr()) + v_offset;
      ET_CHECK_MSG(
          internal::set_tensor_data(kv_tensors[index][j], input_addr, kv_tensors[index][j].nbytes()) == Error::Ok,
          "Failed to set input tensor when updating kv_cache");

      // outputs
      char *output_addr = static_cast<char*>(kv_outputs[index][j].mutable_data_ptr()) + v_offset;
      ET_CHECK_MSG(
          internal::set_tensor_data(kv_outputs[index][j], output_addr, kv_outputs[index][j].nbytes()) == Error::Ok,
          "Failed to set output tensor when updating kv_cache");
      ET_CHECK_MSG(
          modules_[i]->set_output_data_ptr(kv_outputs[index][j], j+1) == Error::Ok,
          "Failed to set output tensor for llama block");
    }
  }

  // affine module
  inputs = {hidden_states};
  Result<std::vector<EValue>> logits_outputs_res = modules_[modules_.size()-1]->forward(inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(logits_outputs_res.error());

  // Bump start_pos by 1
  start_pos.mutable_data_ptr<int32_t>()[0]++;

  // update atten_mask
  atten_mask.mutable_data_ptr<float>()[atten_mask.numel() - 1 - start_pos.const_data_ptr<int32_t>()[0]] = 0;

  return logits_outputs_res.get()[0].toTensor();
}
// TODO: add overloaded method for on-device tokenize
Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::vector<std::vector<ManagedTensor>>& managed_kv_inputs,
    std::vector<std::vector<float>>& freqs_inputs,
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

  std::vector<float> atten_mask_data(max_seq_len_);
  std::fill(atten_mask_data.begin(), atten_mask_data.end()-1, -255.0);
  atten_mask_data.back() = 0;

  std::vector<float> freqs_cos_data(head_dim_/2);
  std::fill(freqs_cos_data.begin(), freqs_cos_data.end(), 0.0);

  std::vector<float> freqs_sin_data(head_dim_/2);
  std::fill(freqs_sin_data.begin(), freqs_sin_data.end(), 0.0);

  std::vector<exec_aten::SizesType> freqs_cos_shape = {1, head_dim_/2};

  std::vector<exec_aten::SizesType> freqs_sin_shape = {1, head_dim_/2};

  std::vector<exec_aten::SizesType> atten_mask_shape = {1, max_seq_len_};

  std::vector<exec_aten::SizesType> logits_data_shape = {1, vocab_size_};

  // initialize tensor wrappers
  ManagedTensor managed_token(
      token_data.data(), 128, token_shape, ScalarType::Int);
  ManagedTensor managed_pos_id(
      start_pos_data.data(), 128, start_pos_shape, ScalarType::Int);
  ManagedTensor managed_atten_mask(
      atten_mask_data.data(), 128, atten_mask_shape, ScalarType::Float);
  ManagedTensor managed_freqs_cos(
      freqs_cos_data.data(), 128, freqs_cos_shape, ScalarType::Float);
  ManagedTensor managed_freqs_sin(
      freqs_sin_data.data(), 128, freqs_sin_shape, ScalarType::Float);


  Tensor token = managed_token.get_aliasing_tensor();
  Tensor atten_mask = managed_atten_mask.get_aliasing_tensor();
  Tensor start_pos = managed_pos_id.get_aliasing_tensor();
  Tensor freqs_cos = managed_freqs_cos.get_aliasing_tensor();
  Tensor freqs_sin = managed_freqs_sin.get_aliasing_tensor();

  // embedding
  std::vector<float> embedding_logits_data(vocab_size_);
  ManagedTensor embedding_managed_logits(
      embedding_logits_data.data(), 128, logits_data_shape, ScalarType::Float);
  Tensor embedding_logits = embedding_managed_logits.get_aliasing_tensor();
  ET_CHECK_MSG(
        modules_[0]->set_output_data_ptr(embedding_logits, 0) == Error::Ok,
        "Failed to set output tensor for embedding module - logits");

  // llama block
  std::vector<std::vector<float>> llama_block_logit_tensor_data(modules_.size()-2);
  std::vector<ManagedTensor> llama_block_logit_tensors, kv_outputs_managed;
  std::vector<std::vector<Tensor>> kv_tensors(modules_.size()-2), kv_outputs(modules_.size()-2);
  std::vector<Result<MethodMeta>> methods_meta = get_methods_meta();

  for (int i = 1; i < modules_.size() - 1; ++i){
    Result<MethodMeta> &cur_meta = methods_meta[i];
    std::vector<float> logits_data(vocab_size_);
    llama_block_logit_tensor_data.push_back(logits_data);
    llama_block_logit_tensors.emplace_back(ManagedTensor(
      logits_data.data(), 128, logits_data_shape, ScalarType::Float));
    Tensor logits = llama_block_logit_tensors.back().get_aliasing_tensor();
    const int k_caches_end = managed_kv_inputs[i-1].size()/2;

    // k caches init
    for (int j = 0; j < k_caches_end; ++j) {
      kv_tensors[i-1].push_back(managed_kv_inputs[i-1][j].get_aliasing_tensor());
      Result<TensorInfo> out_tensor_meta = cur_meta->output_tensor_meta(j+1);
      auto tensor_shape = out_tensor_meta->sizes();
      std::vector<exec_aten::SizesType> out_tensor_shape(
          tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

      int output_offset = (out_tensor_meta->nbytes()+kv_tensors[i-1][j].element_size()) * (max_seq_len_-1);
      char *output_addr = static_cast<char*>(kv_tensors[i-1][j].mutable_data_ptr()) + output_offset;

      kv_outputs_managed.push_back(ManagedTensor(
          output_addr, 128, out_tensor_shape, kv_tensors[i-1][j].scalar_type()));
      kv_outputs[i-1].push_back(kv_outputs_managed.back().get_aliasing_tensor());
      ET_CHECK_MSG(
          modules_[i]->set_output_data_ptr(kv_outputs[i-1][j], j+1) == Error::Ok,
          "Failed to set output tensor for llama block");
    }
    // v caches init
    for (int j = k_caches_end; j < managed_kv_inputs[i-1].size(); ++j) {
      kv_tensors[i-1].push_back(managed_kv_inputs[i-1][j].get_aliasing_tensor());
      char *output_addr = static_cast<char*>(kv_tensors[i-1][j].mutable_data_ptr()) +
          (max_seq_len_-1)*head_dim_*kv_tensors[i-1][j].element_size();

      Result<TensorInfo> out_tensor_meta = cur_meta->output_tensor_meta(j+1);
      auto tensor_shape = out_tensor_meta->sizes();
      std::vector<exec_aten::SizesType> out_tensor_shape(
          tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

      kv_outputs_managed.push_back(ManagedTensor(
          output_addr, 128, out_tensor_shape, kv_tensors[i-1][j].scalar_type()));
      kv_outputs[i-1].push_back(kv_outputs_managed.back().get_aliasing_tensor());
      ET_CHECK_MSG(
          modules_[i]->set_output_data_ptr(kv_outputs[i-1][j], j+1) == Error::Ok,
          "Failed to set output tensor for llama block");
    }
    ET_CHECK_MSG(
        modules_[i]->set_output_data_ptr(logits, 0) == Error::Ok,
        "Failed to set output tensor for llama block - logits");
  }

  // affine layer
  std::vector<float> affine_logits_data(vocab_size_);
  ManagedTensor affine_managed_logits(
      affine_logits_data.data(), 128, logits_data_shape, ScalarType::Float);
  Tensor affine_logits = affine_managed_logits.get_aliasing_tensor();
  ET_CHECK_MSG(
        modules_[modules_.size()-1]->set_output_data_ptr(affine_logits, 0) == Error::Ok,
        "Failed to set output tensor for affine module - logits");

  // Start consuming user's prompts and generating new tokens
  std::string final_output;
  while (pos < seq_len - 1) {
    for(int i = 0; i < head_dim_/2; i++){
      freqs_cos.mutable_data_ptr<float>()[i] = freqs_inputs[0][pos*(head_dim_/2)+i];
      freqs_sin.mutable_data_ptr<float>()[i] = freqs_inputs[1][pos*(head_dim_/2)+i];
    }

    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(cur_token, token, start_pos, atten_mask, freqs_cos, freqs_sin, kv_tensors, kv_outputs);
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

std::vector<Result<MethodMeta>> Runner::get_methods_meta() {
  std::vector<Result<MethodMeta>> tmp;
  for (auto& module : modules_){
    tmp.push_back(module->method_meta("forward"));
  }
  return tmp;
}

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
  Module* module,
    std::string method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
  Module* module,
    std::string method_name,
    bool default_val);

} // namespace executor
} // namespace torch
