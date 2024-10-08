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
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/executor/method_meta.h>

#if defined(QAIHUB_LLAMA3_RUNNER)
#define QAIHUB_LLAMA_NUM_HEADS 8
#define QAIHUB_LLAMA_LOGITS 128256
#else
#define QAIHUB_LLAMA_NUM_HEADS 32
#define QAIHUB_LLAMA_LOGITS 32000
#endif

namespace example {

class Memory {
 public:
  Memory(
      const std::vector<std::string>& pos_embs_path,
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules);
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
  std::vector<std::string> pos_embs_path_;
  std::vector<std::shared_ptr<executorch::extension::Module>> modules_;
};

class BertMemory : public Memory {
 public:
  BertMemory(
      const std::vector<std::string>& pos_embs_path,
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules,
      std::vector<int> shard_layers);
  void prepare_io(const std::vector<executorch::runtime::Result<
                      executorch::runtime::MethodMeta>>& methods_meta) override;
  void update_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  struct IO {
    int32_t input_ids[1024 * 2];
    uint16_t hidden_state[1024 * 4096];
    uint16_t attention_mask[1024 * 1024];
    uint16_t position_ids_cos[1024 * 64];
    uint16_t position_ids_sin[1024 * 64];
    uint8_t k_cache[32][QAIHUB_LLAMA_NUM_HEADS][128 * 1024];
    uint8_t v_cache[32][QAIHUB_LLAMA_NUM_HEADS][1024 * 128];
    uint16_t logits[QAIHUB_LLAMA_LOGITS];
  };

 private:
  std::unique_ptr<executorch::aten::TensorImpl> input_ids_;
  std::unique_ptr<executorch::aten::TensorImpl> hidden_state_;
  std::unique_ptr<executorch::aten::TensorImpl> attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> position_ids_cos_;
  std::unique_ptr<executorch::aten::TensorImpl> position_ids_sin_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_;
  std::unique_ptr<executorch::aten::TensorImpl> logits_;
  std::vector<int> shard_layers_;
  int num_heads_;
};

class ThreadPool {
 public:
  ThreadPool();
  ~ThreadPool();

  std::future<void> issue(std::function<void(void*)> func, void* arg);
  size_t num_workers();

 private:
  struct JobInfo {
    explicit JobInfo(std::packaged_task<void(void*)>&& func, void* arg)
        : func(std::move(func)), arg(arg) {}
    explicit JobInfo(JobInfo&& job_info)
        : func(std::move(job_info.func)), arg(job_info.arg) {}
    std::packaged_task<void(void*)> func;
    void* arg;
  };
  size_t num_workers_;
  std::vector<std::thread> threads_;
  std::queue<JobInfo> jobs_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

class KVCachedMemory : public Memory {
 public:
  KVCachedMemory(
      const std::vector<std::string>& pos_embs_path,
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules,
      std::vector<int> shard_layers);
  void prepare_io(const std::vector<executorch::runtime::Result<
                      executorch::runtime::MethodMeta>>& methods_meta) override;
  void update_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  struct IO {
    int32_t input_ids;
    uint16_t hidden_state[4096];
    uint16_t attention_mask[1024];
    uint16_t position_ids_cos[1024 * 64];
    uint16_t position_ids_sin[1024 * 64];
    uint8_t k_cache[32][QAIHUB_LLAMA_NUM_HEADS][129 * 1023];
    uint8_t v_cache[32][(QAIHUB_LLAMA_NUM_HEADS + 1) * 1023 * 128];
    uint8_t k_cache_out[32][QAIHUB_LLAMA_NUM_HEADS][128];
    uint16_t logits[QAIHUB_LLAMA_LOGITS];
  };
  struct LoopRange {
    int32_t start;
    int32_t end;
    int32_t step;
  };

 private:
  std::unique_ptr<executorch::aten::TensorImpl> input_ids_;
  std::unique_ptr<executorch::aten::TensorImpl> hidden_state_;
  std::unique_ptr<executorch::aten::TensorImpl> attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> position_ids_cos_;
  std::unique_ptr<executorch::aten::TensorImpl> position_ids_sin_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;
  std::unique_ptr<executorch::aten::TensorImpl> logits_;
  std::vector<LoopRange> lr_update_kv_;
  std::vector<std::future<void>> futures_;
  ThreadPool thread_pool_;
  std::vector<int> shard_layers_;
  int num_heads_;
};

} // namespace example
