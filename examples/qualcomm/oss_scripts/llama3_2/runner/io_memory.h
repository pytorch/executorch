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

#define QNN_LLAMA3_2_LOGITS 128256
#define QNN_LLAMA3_2_SEQLEN 512 // adjustable based on llama export
#define QNN_LLAMA3_2_NUM_HEADS 8

#if defined(LLAMA3_2_3B_RUNNER)
#define QNN_LLAMA3_2_HEAD_DIM 128
#define QNN_LLAMA3_2_NUM_LAYERS 28
#else
#define QNN_LLAMA3_2_HEAD_DIM 64
#define QNN_LLAMA3_2_NUM_LAYERS 16
#endif

namespace example {

class Memory {
 public:
  Memory(std::vector<std::shared_ptr<executorch::extension::Module>>& modules);
  virtual ~Memory();
  virtual void prepare_io(
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

class KVCachedMemory : public Memory {
 public:
  KVCachedMemory(
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules);
  void prepare_io(const std::vector<executorch::runtime::Result<
                      executorch::runtime::MethodMeta>>& methods_meta) override;
  void update_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  struct IO {
    int32_t input_tok;
    int32_t input_pos;
    float attention_mask[QNN_LLAMA3_2_SEQLEN];
    uint8_t k_cache[QNN_LLAMA3_2_NUM_LAYERS][QNN_LLAMA3_2_NUM_HEADS]
                   [(QNN_LLAMA3_2_HEAD_DIM + 1) * (QNN_LLAMA3_2_SEQLEN - 1)];
    uint8_t v_cache[QNN_LLAMA3_2_NUM_LAYERS]
                   [(QNN_LLAMA3_2_NUM_HEADS + 1) * (QNN_LLAMA3_2_SEQLEN - 1) *
                    (QNN_LLAMA3_2_HEAD_DIM)];
    uint8_t k_cache_out[QNN_LLAMA3_2_NUM_LAYERS][QNN_LLAMA3_2_NUM_HEADS]
                       [QNN_LLAMA3_2_HEAD_DIM];
    float logits[QNN_LLAMA3_2_LOGITS];
  };
  struct LoopRange {
    int32_t start;
    int32_t end;
    int32_t step;
  };

 private:
  std::unique_ptr<executorch::aten::TensorImpl> input_tok_;
  std::unique_ptr<executorch::aten::TensorImpl> input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> hidden_state_;
  std::unique_ptr<executorch::aten::TensorImpl> attention_mask_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;
  std::unique_ptr<executorch::aten::TensorImpl> logits_;
  std::vector<int> shard_layers_;
  int num_heads_;
};

} // namespace example
