/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/models/llama2/sampler/sampler.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>

class RpcMemAllocator {
 public:
  RpcMemAllocator(QnnMemDescriptor shared_buffer_type)
      : shared_buffer_type_(shared_buffer_type){};
  bool allocate(size_t bytes, size_t alignment) {
    ptr_ = QnnExecuTorchAllocCustomMem(bytes, alignment);
    if (ptr_ == nullptr) {
      ET_LOG(
          Info,
          "Allocate Rpc mem falied, fallback to nromal ptr: bytes=%zu, alignment=%zu",
          bytes,
          alignment);
      input_data_.resize(bytes);
      ptr_ = input_data_.data();
    }
    return ptr_ != nullptr;
  }

  ~RpcMemAllocator() {
    if (shared_buffer_type_ == QnnMemDescriptor::kIon ||
        shared_buffer_type_ == QnnMemDescriptor::kCustom) {
      if (ptr_ != nullptr) {
        QnnExecuTorchFreeCustomMem(ptr_);
      }
    }
  }

  void* GetPtr() {
    return ptr_;
  }

 private:
  QnnMemDescriptor shared_buffer_type_;
  void* ptr_{nullptr};
  std::vector<char> input_data_;
  std::vector<size_t> tensor_base_addrs_;
};

#define DEFINE_IOMEMMGR_ACCESSOR(name)                  \
  size_t get_##name##_pos() const {                     \
    return name##_pos_;                                 \
  }                                                     \
  char* get_##name##_ptr() const {                      \
    return reinterpret_cast<char*>(ptr_) + name##_pos_; \
  }                                                     \
  char* set_##name##_ptr() {                            \
    CustomMemTensorInfo info = {                        \
        ptr_,                                           \
        ptr_ + name##_pos_,                             \
        name##_pos_,                                    \
        io_info_.name.size,                             \
        io_info_.name.shape.data(),                     \
        io_info_.name.rank,                             \
        io_info_.name.dtype};                           \
    QnnExecuTorchAddCustomMemTensorInfo(info);          \
    return reinterpret_cast<char*>(ptr_) + name##_pos_; \
  }

#define DEFINE_IOMEMMGR_VEC_ACCESSOR(name)                   \
  const std::vector<size_t>& get_##name##_pos_vec() const {  \
    return name##_pos_;                                      \
  }                                                          \
  char* get_##name##_ptr(int idx) {                          \
    return ptr_ + name##_pos_[idx];                          \
  }                                                          \
  char* set_##name(int idx, size_t pos) {                    \
    name##_pos_[idx] = pos;                                  \
    CustomMemTensorInfo info = {                             \
        ptr_,                                                \
        ptr_ + name##_pos_[idx],                             \
        name##_pos_[idx],                                    \
        io_info_.name.size,                                  \
        io_info_.name.shape.data(),                          \
        io_info_.name.rank,                                  \
        io_info_.name.dtype};                                \
    QnnExecuTorchAddCustomMemTensorInfo(info);               \
    return reinterpret_cast<char*>(ptr_) + pos;              \
  }                                                          \
  char* update_##name(int idx, size_t shift_size) {          \
    name##_pos_[idx] += shift_size;                          \
    return reinterpret_cast<char*>(ptr_) + name##_pos_[idx]; \
  }

namespace torch {
namespace executor {
class IoMemMgr {
 public:
  // Allocate a big memory which is capable to contain all IO of all modules
  IoMemMgr(){};
  IoMemMgr(MethodMeta method_meta);

  struct InfoAttrs {
    std::unique_ptr<TensorInfo> tensor_meta;
    size_t size = 0;
    std::vector<uint32_t> shape;
    uint32_t rank;
    size_t element_size;
    torch::executor::ScalarType dtype;
  };

  struct IoInfo {
    InfoAttrs input_token;
    InfoAttrs pos_idx;
    InfoAttrs atten_mask;
    InfoAttrs k_caches_read;
    InfoAttrs k_caches_write;
    InfoAttrs v_caches_read;
    InfoAttrs v_caches_write;
    InfoAttrs logit;
    std::vector<InfoAttrs*> tensor_info{
        &input_token,
        &pos_idx,
        &atten_mask,
        &k_caches_read,
        &k_caches_write,
        &v_caches_read,
        &v_caches_write,
        &logit,
    };
  };

  bool allocate(size_t alignment) {
    bool ret = rpc_mem_allocator.allocate(total_nbytes_, alignment);
    ptr_ = reinterpret_cast<char*>(rpc_mem_allocator.GetPtr());
    return ret;
  }
  bool init_tensors();

  char* get_custom_mem_ptr() {
    return ptr_;
  }

  // Pointers of k cache read, v cache read and write are shifted every step.
  // Set them first to register mem handle during qnn delegation init.
  void set_all_shifted_ptrs(size_t max_seq_len);

  DEFINE_IOMEMMGR_ACCESSOR(atten_mask);
  DEFINE_IOMEMMGR_ACCESSOR(input_token);
  DEFINE_IOMEMMGR_ACCESSOR(pos_idx);
  DEFINE_IOMEMMGR_ACCESSOR(logit);

  DEFINE_IOMEMMGR_VEC_ACCESSOR(k_caches_read);
  DEFINE_IOMEMMGR_VEC_ACCESSOR(k_caches_write);
  DEFINE_IOMEMMGR_VEC_ACCESSOR(v_caches_read);
  DEFINE_IOMEMMGR_VEC_ACCESSOR(v_caches_write);

 private:
  size_t total_nbytes_{0};
  char* ptr_{nullptr};
  void compute_total_nbytes();
  void set_tensor_meta();
  void init_io_info();

  size_t atten_mask_pos_;
  size_t input_token_pos_{0};
  size_t logit_pos_;
  size_t pos_idx_pos_;
  std::vector<size_t> k_caches_read_pos_;
  std::vector<size_t> k_caches_write_pos_;
  std::vector<size_t> v_caches_read_pos_;
  std::vector<size_t> v_caches_write_pos_;

  IoInfo io_info_;
  std::unique_ptr<MethodMeta> method_meta_;
  RpcMemAllocator rpc_mem_allocator{QnnMemDescriptor::kCustom};
  std::unordered_map<ScalarType, size_t> scalar_type_to_size = {
      {ScalarType::Int, sizeof(int32_t)},
      {ScalarType::Float, sizeof(float)},
      {ScalarType::Char, sizeof(int8_t)},
      {ScalarType::Short, sizeof(int16_t)},
      {ScalarType::Byte, sizeof(uint8_t)},
      {ScalarType::Bits16, sizeof(uint16_t)},
  };
};

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f);

  struct Stats {
    // Scaling factor for timestamps - in this case, we use ms.
    const long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
    // Time stamps for the different stages of the execution
    // model_load_start_ms: Start of model loading.
    long model_load_start_ms;
    // model_load_end_ms: End of model loading.
    long model_load_end_ms;
    // inference_start_ms: Immediately after the model is loaded (or we check
    // for model load), measure the inference time.
    long inference_start_ms;
    // prompt_eval_end_ms: Prompt array allocation and tokenization. Ends right
    // before the inference loop starts
    long prompt_eval_end_ms;
    // first_token: Timestamp when the first generated token is emitted
    long first_token_ms;
    // inference_end_ms: End of inference/generation.
    long inference_end_ms;
    // Keep a running total of the time spent in sampling.
    long aggregate_sampling_time_ms;
    // Token count from prompt
    int64_t num_prompt_tokens;
    // Token count from generated (total - prompt)
    int64_t num_generated_tokens;
  };

  bool is_loaded() const;
  Error load();
  Error mem_alloc(size_t alignment, size_t seq_len);
  Error generate(
      const std::string& prompt,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  void stop();
  Result<MethodMeta> get_method_meta();

 private:
  // metadata
  template <typename T>
  T getMetadataHelper(std::string method_name, T default_val);
  template <typename T>
  int32_t logitsToToken(const exec_aten::Tensor& logits_tensor);
  Result<torch::executor::Tensor> run_model_step(
      int64_t input_token,
      Tensor& token,
      Tensor& start_pos,
      Tensor& atten_mask,
      std::vector<Tensor>& kv_tensors,
      std::vector<Tensor>& kv_outputs);
  // metadata
  int32_t vocab_size_;
  int64_t bos_id_;
  int64_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  int32_t head_dim_;
  int32_t dim_;
  std::unordered_set<std::string> model_methods_;
  std::unique_ptr<Module> module_;
  std::string tokenizer_path_;
  std::string model_path_;
  float temperature_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
  bool shouldStop_{false};
  Stats stats_;
  IoMemMgr io_mem_mgr_;
};

} // namespace executor
} // namespace torch
