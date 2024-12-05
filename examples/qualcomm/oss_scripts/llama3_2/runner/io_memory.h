/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/executor/method_meta.h>

namespace example {

class Memory {
 public:
  Memory(std::vector<std::shared_ptr<executorch::extension::Module>>& modules);
  virtual ~Memory();
  virtual void prepare_prefill_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) = 0;
  virtual void prepare_kv_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) = 0;
  virtual void update_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors) = 0;
  void* get_mutable_ptr();
  std::vector<executorch::aten::Tensor> get_input_tensors(int shard_index);
  std::vector<executorch::aten::Tensor> get_output_tensors(int shard_index);

 protected:
  std::unique_ptr<void, void (*)(void*)> data_ptr_;
  std::vector<std::vector<executorch::aten::TensorImpl*>> input_tensors_;
  std::vector<std::vector<executorch::aten::TensorImpl*>> output_tensors_;
  std::vector<std::shared_ptr<executorch::extension::Module>> modules_;
};

class HybridMemory : public Memory {
 public:
  HybridMemory(
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules,
      int32_t max_seq_len,
      int32_t vocab_size,
      int32_t num_layers,
      int32_t head_dim,
      int32_t num_heads);
  void prepare_prefill_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void prepare_kv_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void update_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  struct IO {
    int32_t input_tok;
    int32_t input_pos;
    std::vector<float> attention_mask;
    std::vector<std::vector<std::vector<uint8_t>>> k_cache;
    std::vector<std::vector<uint8_t>> v_cache;
    std::vector<std::vector<uint8_t>> k_cache_out;
    std::vector<float> logits;
    std::vector<int32_t> prefill_input_toks;
    std::vector<float> prefill_atten_mask;
    std::vector<float> prefill_logits;
  };

 private:
  std::unique_ptr<executorch::aten::TensorImpl> input_tok_;
  std::unique_ptr<executorch::aten::TensorImpl> input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> hidden_state_;
  std::unique_ptr<executorch::aten::TensorImpl> attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_input_toks_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_attn_mask_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;
  std::unique_ptr<executorch::aten::TensorImpl> logits_;
  std::vector<int> shard_layers_;
  int32_t max_seq_len_;
  int32_t vocab_size_;
  int32_t num_layers_;
  int32_t head_dim_;
  int32_t num_heads_;
};

} // namespace example
