/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/qualcomm/oss_scripts/llama2/runner/runner.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <ctime>
#include <memory>
#include <sstream>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::BPETokenizer;
using executorch::extension::llm::Sampler;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

// TODO: Remove this usage of an internal-only function.
using executorch::runtime::internal::set_tensor_data;

namespace example {

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
          Module::LoadMode::MmapUseMlockIgnoreErrors)),
      tokenizer_path_(tokenizer_path),
      model_path_(model_path),
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
  stats_.model_load_start_ms = time_in_ms();
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
  head_dim_ = getMetadataHelper<int64_t>("get_head_dim", 32);
  dim_ = getMetadataHelper<int64_t>("get_dim", 4096);

  // Load tokenizer
  tokenizer_ = std::make_unique<BPETokenizer>();
  tokenizer_->load(tokenizer_path_);
  if (tokenizer_->bos_tok() != bos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's BOS id %lu does not match model's BOS id %ld, will override tokenizer's BOS.",
        tokenizer_->bos_tok(),
        bos_id_);
  }
  if (tokenizer_->eos_tok() != eos_id_) {
    ET_LOG(
        Error,
        "Tokenizer's EOS id %lu does not match model's EOS id %ld, will override tokenizer's EOS.",
        tokenizer_->eos_tok(),
        eos_id_);
  }
  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));
  stats_.model_load_end_ms = time_in_ms();

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
int32_t Runner::logitsToToken(const Tensor& logits_tensor) {
  T* logits = logits_tensor.mutable_data_ptr<T>();

  // Since the logits are for all tokens, get the last token probabilities
  T* logits_last = logits;
  return sampler_->sample(logits_last);
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<Tensor> Runner::run_model_step(
    int64_t input_token,
    TensorPtr& token,
    TensorPtr& start_pos,
    TensorPtr& atten_mask,
    std::vector<TensorPtr>& kv_tensors,
    std::vector<TensorPtr>& kv_outputs) {
  token->mutable_data_ptr<int32_t>()[0] = input_token;

  // inputs:[tokens, start_pos, atten_mask, k_cache, v_cache]
  auto outputs_res = module_->forward({token, start_pos, atten_mask});
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());

  // TODO: need to handle batch size != 1
  size_t v_offset = kv_outputs[0]->nbytes();
  size_t el_size = kv_outputs[0]->element_size();
  size_t k_input_step = (max_seq_len_ - 1) * el_size;
  int k_tensors_end = kv_tensors.size() / 2;
  // update k caches
  for (int j = 0; j < k_tensors_end; ++j) {
    uint8_t* input_addr =
        static_cast<uint8_t*>(kv_tensors[j]->mutable_data_ptr());
    uint8_t* output_addr =
        static_cast<uint8_t*>(kv_outputs[j]->mutable_data_ptr());
    // fill the output k values back
    for (int src = 0, dst = k_input_step; src < kv_outputs[j]->nbytes();
         src += el_size, dst += k_input_step) {
      input_addr[dst] = output_addr[src];
    }
    char* new_inp_addr = io_mem_mgr_.update_k_caches_read(j, el_size);
    // inputs
    ET_CHECK_MSG(
        set_tensor_data(
            *kv_tensors[j], new_inp_addr, kv_tensors[j]->nbytes()) == Error::Ok,
        "Failed to set input tensor when updating k_cache");
  }
  // update v caches
  for (int j = k_tensors_end, v_idx = 0; j < kv_tensors.size(); ++j, ++v_idx) {
    // inputs
    char* new_inp_addr = io_mem_mgr_.update_v_caches_read(v_idx, v_offset);

    ET_CHECK_MSG(
        set_tensor_data(
            *kv_tensors[j], new_inp_addr, kv_tensors[j]->nbytes()) == Error::Ok,
        "Failed to set input tensor when updating v_cache");
    // outputs
    char* new_out_addr = io_mem_mgr_.update_v_caches_write(v_idx, v_offset);
    ET_CHECK_MSG(
        set_tensor_data(
            *kv_outputs[j], new_out_addr, kv_outputs[j]->nbytes()) == Error::Ok,
        "Failed to set output tensor when updating v_cache");
    ET_CHECK_MSG(
        module_->set_output(*kv_outputs[j], j + 1) == Error::Ok,
        "Failed to set llama output data pointer");
  }

  // Bump start_pos by 1
  start_pos->mutable_data_ptr<int32_t>()[0]++;

  // update atten_mask
  atten_mask->mutable_data_ptr<float>()
      [atten_mask->numel() - 1 - start_pos->const_data_ptr<int32_t>()[0]] = 0;
  return outputs_res.get()[0].toTensor();
}
// TODO: add overloaded method for on-device tokenize
Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  ET_CHECK_MSG(is_loaded(), "Please invoke load method first");

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.
  stats_.inference_start_ms = time_in_ms();
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
  std::vector<SizesType> token_shape = {1, 1};

  io_mem_mgr_.get_input_token_ptr()[0] = 0;
  std::vector<SizesType> start_pos_shape = {1, 1};

  float* atten_mask_ptr =
      reinterpret_cast<float*>(io_mem_mgr_.get_atten_mask_ptr());
  std::fill(atten_mask_ptr, atten_mask_ptr + max_seq_len_, -255);
  atten_mask_ptr[max_seq_len_ - 1] = 0;

  std::vector<SizesType> atten_mask_shape = {1, max_seq_len_};

  std::vector<SizesType> logits_data_shape = {1, vocab_size_};

  std::vector<SizesType> hidden_states_data_shape = {1, 1, dim_};

  // initialize tensor wrappers
  auto token = from_blob(
      io_mem_mgr_.get_input_token_ptr(), token_shape, ScalarType::Int);
  auto start_pos = from_blob(
      io_mem_mgr_.get_pos_idx_ptr(), start_pos_shape, ScalarType::Int);
  auto atten_mask = from_blob(
      io_mem_mgr_.get_atten_mask_ptr(), atten_mask_shape, ScalarType::Float);

  std::vector<TensorPtr> kv_tensors, kv_outputs;

  Result<MethodMeta> method_meta = get_method_meta();
  size_t num_inputs = method_meta->num_inputs();
  int k_caches_num = (num_inputs - 3) / 2;

  // TODO: need to handle batch size != 1
  // k caches init
  for (int input_index = 3, i = 0; input_index < k_caches_num + 3;
       ++input_index, ++i) {
    // inputs
    Result<TensorInfo> tensor_meta =
        method_meta->input_tensor_meta(input_index);

    auto tensor_shape = tensor_meta->sizes();
    std::vector<SizesType> sizes(
        tensor_shape.data(), tensor_shape.data() + tensor_shape.size());
    kv_tensors.emplace_back(from_blob(
        io_mem_mgr_.get_k_caches_read_ptr(i),
        sizes,
        tensor_meta->scalar_type()));

    // outpus
    Result<TensorInfo> out_tensor_meta = method_meta->output_tensor_meta(i + 1);
    tensor_shape = out_tensor_meta->sizes();
    sizes = std::vector<SizesType>{
        tensor_shape.data(), tensor_shape.data() + tensor_shape.size()};
    kv_outputs.emplace_back(from_blob(
        io_mem_mgr_.get_k_caches_write_ptr(i),
        sizes,
        kv_tensors.back()->scalar_type()));
    ET_CHECK_MSG(
        module_->set_output(kv_outputs.back(), i + 1) == Error::Ok,
        "Failed to set output tensor for kv cache");
  }

  // v caches init
  for (int i = 0, input_index = k_caches_num + 3; input_index < num_inputs;
       ++input_index, ++i) {
    int output_index = i + k_caches_num + 1;
    // inputs
    Result<TensorInfo> tensor_meta =
        method_meta->input_tensor_meta(input_index);
    auto tensor_shape = tensor_meta->sizes();
    std::vector<SizesType> sizes(
        tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

    kv_tensors.emplace_back(from_blob(
        io_mem_mgr_.get_v_caches_read_ptr(i),
        sizes,
        tensor_meta->scalar_type()));

    // outputs
    Result<TensorInfo> out_tensor_meta =
        method_meta->output_tensor_meta(output_index);
    tensor_shape = out_tensor_meta->sizes();
    sizes = std::vector<SizesType>{
        tensor_shape.data(), tensor_shape.data() + tensor_shape.size()};

    kv_outputs.push_back(from_blob(
        io_mem_mgr_.get_v_caches_write_ptr(i),
        sizes,
        kv_tensors.back()->scalar_type()));
    ET_CHECK_MSG(
        module_->set_output(kv_outputs.back(), output_index) == Error::Ok,
        "Failed to set output tensor for llama block");
  }

  auto affine_logits = from_blob(
      reinterpret_cast<float*>(io_mem_mgr_.get_logit_ptr()),
      logits_data_shape,
      ScalarType::Float);
  ET_CHECK_MSG(
      module_->set_output(affine_logits) == Error::Ok,
      "Failed to set output tensor for affine module - logits");

  // Start consuming user's prompts and generating new tokens
  std::string final_output;
  while (pos < seq_len - 1) {
    // Run the model
    auto logits_res = run_model_step(
        cur_token, token, start_pos, atten_mask, kv_tensors, kv_outputs);
    if (pos == num_prompt_tokens) {
      stats_.first_token_ms = time_in_ms();
    } else if (pos == num_prompt_tokens - 1) {
      stats_.prompt_eval_end_ms = time_in_ms();
    }

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    Tensor& logits_tensor = logits_res.get();
    prev_token = cur_token;
    long sample_start_time_ms = time_in_ms();

    cur_token = logitsToToken<float>(logits_tensor);
    stats_.aggregate_sampling_time_ms += time_in_ms() - sample_start_time_ms;

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
  stats_.inference_end_ms = time_in_ms();

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

IoMemMgr::IoMemMgr(MethodMeta method_meta) {
  method_meta_ = std::make_unique<MethodMeta>(method_meta);
  init_io_info();
  compute_total_nbytes();
}

void IoMemMgr::init_io_info() {
  set_tensor_meta();
  for (auto info : io_info_.tensor_info) {
    info->size = info->tensor_meta->nbytes();
    info->rank = info->tensor_meta->sizes().size();
    info->shape.resize(info->rank);
    for (int i = 0; i < info->rank; i++) {
      info->shape[i] =
          static_cast<uint32_t>(info->tensor_meta->sizes().data()[i]);
    }
    info->dtype = info->tensor_meta->scalar_type();
    info->element_size = scalar_type_to_size[info->tensor_meta->scalar_type()];
  }
};

void IoMemMgr::set_tensor_meta() {
  io_info_.input_token.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->input_tensor_meta(0).get());
  io_info_.pos_idx.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->input_tensor_meta(1).get());
  io_info_.atten_mask.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->input_tensor_meta(2).get());

  io_info_.k_caches_read.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->input_tensor_meta(3).get());
  io_info_.k_caches_write.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->output_tensor_meta(1).get());

  io_info_.v_caches_read.tensor_meta = std::make_unique<TensorInfo>(
      method_meta_->input_tensor_meta(method_meta_->num_inputs() - 1).get());
  io_info_.v_caches_write.tensor_meta = std::make_unique<TensorInfo>(
      method_meta_->output_tensor_meta(method_meta_->num_outputs() - 1).get());

  io_info_.logit.tensor_meta =
      std::make_unique<TensorInfo>(method_meta_->output_tensor_meta(0).get());
}

void IoMemMgr::compute_total_nbytes() {
  total_nbytes_ = io_info_.input_token.size + io_info_.pos_idx.size +
      io_info_.atten_mask.size + io_info_.logit.size;
  size_t num_heads = (method_meta_->num_inputs() - 3) / 2;

  // To update v cache via shifting pointer, v caches need a buffer with size
  // of (max_seq_len_ - 1) * head_dim_. It is equivalent to one more cache
  size_t num_v_cache = num_heads + 1;
  // To update v cache via shifting pointer, k buffer need the size of
  // max_seq_len - 1
  size_t k_buffer = io_info_.k_caches_read.size / io_info_.k_caches_write.size;

  // k_caches_read need a buffer with size of head_dim_
  total_nbytes_ += num_heads * io_info_.k_caches_read.size + k_buffer;
  total_nbytes_ += num_heads * io_info_.k_caches_write.size;
  total_nbytes_ += num_v_cache * io_info_.v_caches_read.size;
  // Add a head dim size for the convinience of shifting ptr from the last
  // non-used v cache write
  total_nbytes_ += io_info_.v_caches_write.size;
}

bool IoMemMgr::init_tensors() {
  size_t cur_pos = input_token_pos_;
  pos_idx_pos_ = cur_pos += io_info_.input_token.size;
  atten_mask_pos_ = cur_pos += io_info_.pos_idx.size;
  logit_pos_ = cur_pos += io_info_.atten_mask.size;
  set_input_token_ptr();
  set_pos_idx_ptr();
  set_atten_mask_ptr();
  set_logit_ptr();

  // set start point of kv caches
  cur_pos += io_info_.logit.size;

  size_t num_heads = (method_meta_->num_inputs() - 3) / 2;
  k_caches_read_pos_.resize(num_heads);
  k_caches_write_pos_.resize(num_heads);
  v_caches_read_pos_.resize(num_heads);
  v_caches_write_pos_.resize(num_heads);

  for (int i = 0; i < num_heads; i++) {
    set_k_caches_read(i, cur_pos);
    cur_pos += io_info_.k_caches_read.size;
  }
  // add a size of k caches buffer
  cur_pos += io_info_.k_caches_read.size / io_info_.k_caches_write.size;
  for (int i = 0; i < num_heads; i++) {
    set_k_caches_write(i, cur_pos);
    cur_pos += io_info_.k_caches_write.size;
  }

  for (int i = 0; i < num_heads; i++) {
    set_v_caches_read(i, cur_pos);
    set_v_caches_write(i, cur_pos + io_info_.v_caches_read.size);
    cur_pos += io_info_.v_caches_read.size;
  }
  // add a caches as the b caches buffer
  cur_pos += io_info_.v_caches_read.size;
  return cur_pos <= total_nbytes_;
}

void IoMemMgr::set_all_shifted_ptrs(size_t seq_len) {
  auto iter_setter = [&](std::vector<size_t>& cache,
                         size_t shift_size,
                         InfoAttrs& tensor_info) {
    for (int i = 0; i < cache.size(); ++i) {
      size_t pos = cache[i] + shift_size;
      CustomMemTensorInfo info = {
          ptr_,
          ptr_ + pos,
          pos,
          tensor_info.size,
          tensor_info.shape.data(),
          tensor_info.rank,
          tensor_info.dtype};
      QnnExecuTorchAddCustomMemTensorInfo(info);
    }
  };
  for (int i = 0; i < seq_len; ++i) {
    iter_setter(
        k_caches_read_pos_,
        i * io_info_.k_caches_read.element_size,
        io_info_.k_caches_read);
    iter_setter(
        v_caches_read_pos_,
        i * io_info_.v_caches_write.size,
        io_info_.v_caches_read);
    iter_setter(
        v_caches_write_pos_,
        i * io_info_.v_caches_write.size,
        io_info_.v_caches_write);
  }
}

void Runner::stop() {
  shouldStop_ = true;
}

Result<MethodMeta> Runner::get_method_meta() {
  return module_->method_meta("forward");
}

Error Runner::mem_alloc(size_t alignment, size_t seq_len) {
  Result<MethodMeta> method_meta_result = get_method_meta();
  io_mem_mgr_ = IoMemMgr(method_meta_result.get());
  ET_CHECK_MSG(
      io_mem_mgr_.allocate(alignment),
      "IoMemMgr failed to allocate custom memory");

  ET_CHECK_MSG(
      io_mem_mgr_.init_tensors(),
      "IoMemMgr required more bytes than allocated bytes");

  io_mem_mgr_.set_all_shifted_ptrs(seq_len);
  // To register rpc_mem_handle from SharedBuffer
  // Reset and re-init again to trigger registered function
  module_.reset();
  module_ = std::make_unique<Module>(
      model_path_, Module::LoadMode::MmapUseMlockIgnoreErrors),
  ET_CHECK_MSG(load() == Error::Ok, "Runner failed to load method");

  return Error::Ok;
}

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
    std::string method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
    std::string method_name,
    bool default_val);

} // namespace example
