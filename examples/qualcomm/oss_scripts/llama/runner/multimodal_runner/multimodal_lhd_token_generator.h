/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>

namespace example {

/**
 * @class MultimodalLhdTokenGenerator
 * @brief Extended LhdTokenGenerator with multimodal embedding support
 */
template <typename T>
class MultimodalLhdTokenGenerator
    : public example::MultimodalTokenGenerator<T> {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
    int32_t ngram;
    int32_t window;
    int32_t gcap;
    int sliding_window;
    CacheMode cache_mode;
    int32_t embedding_dim = 0;
  };
  MultimodalLhdTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      EmbeddingProcessor* embedding_runner,
      DecoderRunner* decoder_runner,
      KVManager<T>* kv_manager,
      const std::string& forward_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats)
      : MultimodalTokenGenerator<T>(
            tokenizer,
            embedding_runner,
            decoder_runner,
            kv_manager,
            forward_name,
            std::move(eos_ids),
            typename MultimodalTokenGenerator<T>::Metadata{
                metadata.context_len,
                metadata.num_heads,
                metadata.num_layers,
                metadata.ar_len,
                metadata.vocab_size,
                metadata.use_int64_token,
                metadata.sliding_window,
                metadata.cache_mode,
                metadata.embedding_dim},
            stats),
        embedding_runner_(embedding_runner),
        metadata_(metadata),
        lhd_branch_(metadata.ngram - 1, std::vector<int32_t>(metadata.window)),
        lhd_branch_prev_(metadata.window),
        ngrams_pool_(metadata.vocab_size, metadata.ngram, metadata.gcap) {
    ET_LOG(
        Info,
        "Use Lookahead decoding: ngram=%d, window=%d, gcap=%d",
        metadata.ngram,
        metadata.window,
        metadata.gcap);

    // initialize position offset
    position_offset_ = std::vector<int32_t>(metadata.ar_len);
    int idx = 0;
    // lookahead branches
    for (int i = 0; i < metadata.ngram - 1; ++i) {
      for (int j = 0; j < metadata.window; ++j) {
        position_offset_[idx++] = i + j;
      }
    }
    // verification branches
    for (int i = 0; i < metadata.gcap; ++i) {
      for (int j = 1; j < metadata.ngram; ++j) {
        position_offset_[idx++] = j;
      }
    }
  }

  ~MultimodalLhdTokenGenerator() = default;

  /**
     * @brief Generate tokens with lookahead decoding.
     * @param tokens Vector of input tokens.
     * @param start_pos Starting position for generation.
     * @param seq_len Length of the sequence to generate.
     * @param token_callback Callback function for generated tokens.
     * @return The number of tokens generated.
     */
  executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits) override;

 private:
  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param cur_token Current token.
   * @param start_pos Starting position.
   */
  void prepare_io(
      std::vector<uint64_t> input_tokens,
      std::vector<int32_t> input_pos);
  void init_attention_mask(int32_t n_past);
  void init_lookahead_branch(const std::vector<uint64_t>& tokens);
  void init_verification_branch(uint64_t cur_token);
  void update_lookahead_branch(const executorch::aten::Tensor& logits_tensor);
  void update_ngrams_pool();

  // Additional members specific to multimodal
  EmbeddingProcessor* embedding_runner_;

  struct NgramData {
    bool active = false;
    int32_t seq_id = -1;

    // match pos
    std::vector<int> i_batch;
    std::vector<int32_t> tokens;
  };

  // n-gram pool
  struct NgramContainer {
    NgramContainer(int n_vocab, int n, int g) {
      cnt.resize(n_vocab);
      head.resize(n_vocab);
      tokens.resize(n_vocab * g * (n - 1));
    }

    int n_total = 0;

    std::vector<size_t> cnt;
    std::vector<int> head;

    // [n_vocab][G][N - 1]
    // for each token of the vocab, keep a ring-buffer of capacity G of n-grams
    // of size N - 1
    std::vector<int32_t> tokens;
  };

  Metadata metadata_;

  // lookahead branch
  bool is_lhd_branch_initialized_{false};
  // [N - 1][W]
  std::vector<std::vector<int32_t>> lhd_branch_;
  // [W]
  std::vector<int32_t> lhd_branch_prev_;

  // verification branch
  std::vector<NgramData> v_branch_;

  // position offset in attention mask
  std::vector<int32_t> position_offset_;

  // n-gram pools
  NgramContainer ngrams_pool_;
};
} // namespace example
