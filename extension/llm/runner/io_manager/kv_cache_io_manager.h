/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * @brief Drives a decoder step that takes its KV cache as I/O ("kvcache_io").
 *
 * The program's method is
 *
 *   (input_tokens, atten_mask, pos_ids, past_k_0..N-1, past_v_0..N-1)
 *       -> (logits, new_k_0..N-1, new_v_0..N-1)
 *
 * with `past_k_i` stored transposed `[1, H, D, P]`, `past_v_i` `[1, H, P, D]`,
 * (an encoder-decoder's step also takes `cross_k_i` / `cross_v_i`, fixed per
 * input and written by its encoder method, after the past caches),
 * only the new tokens' K/V slice returned, and an additive (or, quantized,
 * 0/max) attention mask over the `P` past slots followed by the step's own
 * slots. This IOManager owns the cache and mask buffers: before each step it
 * opens the mask over the filled past slots; after it, it writes the new slice
 * into the past buffers at the current position. The runner above only sees
 * tokens in and logits out.
 */
class ET_EXPERIMENTAL KvCacheIOManager : public IOManager {
 public:
  static constexpr const char* kContract = "kvcache_io";

  explicit KvCacheIOManager(ET_MODULE_NAMESPACE::Module& module);
  ~KvCacheIOManager() override;

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

  ET_NODISCARD runtime::Error load_encoder(
      const std::string& encoder_method) override;

  ET_NODISCARD runtime::Error update_encoder(
      const std::vector<runtime::EValue>& encoder_outputs) override;

 private:
  struct Buffer {
    std::byte* data = nullptr;
    size_t nbytes = 0;
    TensorPtr tensor;
  };
  struct Block {
    void* data;
    bool from_allocator;
  };

  // One method's I/O. Cache buffers are shared by every method with the same
  // past length.
  struct MethodIO {
    int32_t ar_len = 1;
    int32_t past_len = 0;
    int32_t context_len = 0;
    Buffer tokens, mask, positions, logits;
    std::vector<Buffer> new_k, new_v;
    std::vector<Buffer*> past_k, past_v;
    executorch::aten::ScalarType mask_dtype;
    int32_t valid = 0; // tokens of the last step that were real (not padding)
    int64_t start = 0;
  };

  runtime::Error load_method(const std::string& method);
  runtime::Result<std::vector<runtime::EValue>> prepare(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& method);
  runtime::Error update(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& method);
  // Places each buffer in one block from the I/O buffer allocator (the heap if
  // there is none or it declines).
  void allocate(
      const std::vector<std::pair<Buffer*, ::executorch::runtime::TensorInfo>>&
          buffers);
  void fill_mask(MethodIO& io, int64_t n_past, int32_t n_valid);
  void rearrange(int32_t past_len);

  ET_MODULE_NAMESPACE::Module& module_;
  std::unique_ptr<IOBufferAllocator> allocator_;
  std::vector<Block> blocks_;
  std::unordered_map<std::string, MethodIO> methods_;
  // Past caches by past length, then layer.
  std::unordered_map<int32_t, std::vector<Buffer>> past_k_, past_v_;
  int32_t num_layers_ = 0;
  int32_t num_cross_layers_ = 0;
  // Cross-attention keys then values, shared by every method.
  std::vector<Buffer> cross_;
  // Positions filled in the cache, and the past length of the layout that
  // holds them (methods with different step sizes have different layouts).
  int64_t filled_ = 0;
  int32_t active_past_len_ = -1;
  std::vector<float> logits_f_;
  TensorPtr logits_f_tensor_;
  double logits_scale_ = 1.0;
  int64_t logits_zero_point_ = 0;
};

/// Registers the IOManagers this library implements (e.g. "kvcache_io") with
/// `io_manager_registry()`. Idempotent.
void register_builtin_io_managers();

} // namespace llm
} // namespace extension
} // namespace executorch
