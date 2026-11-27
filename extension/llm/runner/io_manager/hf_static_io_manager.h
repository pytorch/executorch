/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/tensor/tensor.h>

#include <map>

namespace executorch {
namespace extension {
namespace llm {

/**
 * @brief Drives Hugging Face's static-cache ABI ("hf_static").
 *
 * Each method is HF's decode graph at a static query length N. Inputs:
 * input_ids [1, N], attention_mask(s) [1, 1, N, L] (bool), position_ids
 * [1, N], then per layer keys and values [1, H, L, D] and cumulative_length.
 * Outputs: the caches and counters updated in place, logits [1, N, V], and
 * optionally the duplicate HF user cache state.
 * A program may carry `prefill` (N tokens per step, for the prompt) and
 * `decode` (one token), or a single `forward`.
 *
 * Version 1 supports float32 and float16 caches. See the versioned contract
 * in docs/source/llm-hf-static-io-contract.md for metadata and padding rules.
 *
 * Sliding-window layers may hold window-sized caches (HF's): get_max_seq_len
 * defines the context, and a window's mask also marks a prompt chunk's padding
 * rows (all false), so the program shifts the window by the real rows only.
 *
 * This IOManager owns the caches and counters, one set shared by every method
 * (from the I/O buffer allocator when there is one, so a delegate reads them in
 * place), and builds each step's tokens, masks and positions as HF's generator
 * does. The program writes each step's rows and counters into those buffers.
 */
class ET_EXPERIMENTAL HfStaticIOManager : public IOManager {
 public:
  static constexpr const char* kContract = "hf_static";
  static constexpr int64_t kContractVersion = 1;

  explicit HfStaticIOManager(ET_MODULE_NAMESPACE::Module& module);
  ~HfStaticIOManager() override;

  ET_NODISCARD runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  ET_NODISCARD runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& prefill_method) override;

  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& decode_method) override;

  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& prefill_method) override;

  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) override;

  runtime::Result<executorch::aten::Tensor> logits(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& method) override;

  std::pair<std::string, std::string> method_names(
      const std::unordered_set<std::string>& available) const override;

  int64_t max_tokens_per_step(const std::string& method) const override;
  int64_t max_tokens_per_step(const std::string& method, int64_t start_pos)
      const override;

 private:
  struct Buffer {
    void* data = nullptr;
    size_t nbytes = 0;
    TensorPtr tensor;
  };
  // A method's per-step inputs; the caches are shared.
  struct Method {
    int64_t query_len = 1;
    size_t num_inputs = 0;
    // Input index -> buffer, for input_ids, the masks and position_ids.
    std::map<size_t, Buffer> step_inputs;
    size_t input_ids = 0;
    size_t position_ids = 0;
    std::vector<size_t> masks;
    size_t first_cache = 0;
    // The logits, when the program leaves them to the runtime.
    Buffer logits;
    // The last step's real (unpadded) token count.
    int64_t valid = 0;
  };

  runtime::Error load_metadata();
  runtime::Error load_method(const std::string& method);
  runtime::Result<std::vector<runtime::EValue>> prepare(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& method);
  void allocate(
      const std::vector<std::pair<Buffer*, runtime::TensorInfo>>& buffers);

  ET_MODULE_NAMESPACE::Module& module_;
  std::unique_ptr<IOBufferAllocator> allocator_;
  struct Block {
    void* data;
    bool from_allocator;
  };
  std::vector<Block> blocks_;
  std::map<std::string, Method> methods_;
  // Per layer keys, values, cumulative length; shared by every method.
  std::vector<Buffer> caches_;
  bool metadata_loaded_ = false;
  executorch::aten::ScalarType cache_dtype_ =
      executorch::aten::ScalarType::Float;
  int32_t num_layers_ = 0;
  int64_t context_len_ = 0;
  std::vector<float> logits_f_;
  TensorPtr logits_f_tensor_;
};

void register_hf_static_io_manager();

} // namespace llm
} // namespace extension
} // namespace executorch
