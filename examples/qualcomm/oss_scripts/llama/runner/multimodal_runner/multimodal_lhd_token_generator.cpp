/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_lhd_token_generator.h>
#include <algorithm>
#include <cstdlib>
using executorch::runtime::Result;

namespace example {

template <typename T>
void MultimodalLhdTokenGenerator<T>::prepare_io(
    std::vector<uint64_t> input_tokens,
    std::vector<int32_t> input_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (i < input_tokens.size()) {
      // Prepare pos data
      this->input_pos_.data[i] = input_pos[i];
    }
  }

  // Generate embeddings for the first metadata_.ar_len tokens
  int num_tokens_to_process =
      std::min(static_cast<int>(input_tokens.size()), metadata_.ar_len);
  std::vector<uint64_t> tokens_to_process(
      input_tokens.begin(), input_tokens.begin() + num_tokens_to_process);

  embedding_runner_->prefill(tokens_to_process);
  const TensorStruct<float>& text_embeddings =
      embedding_runner_->get_prompt_embeddings();
  int64_t embedding_dim = text_embeddings.tensor->size(2);

  // Copy embedding to input buffer from the left
  std::memcpy(
      this->input_embedding_.data,
      text_embeddings.data,
      num_tokens_to_process * embedding_dim * sizeof(float));

  // If metadata_.ar_len > input_tokens.size(), initialize remaining part to 0
  if (metadata_.ar_len > num_tokens_to_process) {
    int remaining_tokens = metadata_.ar_len - num_tokens_to_process;
    std::memset(
        this->input_embedding_.data + num_tokens_to_process * embedding_dim,
        0,
        remaining_tokens * embedding_dim * sizeof(float));
  }
}

template <typename T>
void MultimodalLhdTokenGenerator<T>::init_attention_mask(int32_t n_past) {
  std::vector<int32_t> attention_map;
  attention_map.reserve(metadata_.ar_len);
  // Initialize attention mask with current position
  for (int i = 0; i < metadata_.window; ++i) {
    attention_map.push_back(i - 1);
  }
  for (int i = 1; i < metadata_.ngram - 1; ++i) {
    for (int j = 0; j < metadata_.window; ++j) {
      attention_map.push_back((i - 1) * metadata_.window + j);
    }
  }
  for (int g = 0; g < metadata_.gcap; g++) {
    for (int j = 0; j < metadata_.ngram - 1; j++) {
      if (j == 0)
        attention_map.push_back(0);
      else
        attention_map.push_back(
            (metadata_.window + g) * (metadata_.ngram - 1) + j - 1);
    }
  }

  this->kv_manager_->init_attention_mask(
      this->attention_mask_.data, attention_map, metadata_.ar_len, n_past);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    this->kv_manager_->init_attention_mask(
        this->window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        n_past,
        metadata_.sliding_window,
        position_offset_);
  }
}

template <typename T>
void MultimodalLhdTokenGenerator<T>::init_lookahead_branch(
    const std::vector<uint64_t>& tokens) {
  for (int i = 0; i < metadata_.ngram - 1; ++i) {
    for (int j = 0; j < metadata_.window; ++j) {
      // there are different ways to init these tokens
      if (0) {
        // initialize with a sequence of increasing numbers
        lhd_branch_[i][j] = 1000 + j;
      } else {
        // initialize with the random token from prompt
        lhd_branch_[i][j] = tokens[1 + rand() % (tokens.size() - 1)];
      }
    }
  }
  is_lhd_branch_initialized_ = true;
}

template <typename T>
void MultimodalLhdTokenGenerator<T>::init_verification_branch(
    uint64_t cur_token) {
  const int g_cur = ngrams_pool_.cnt[cur_token];

  v_branch_.resize(g_cur);
  for (int g = 0; g < g_cur; g++) {
    v_branch_[g].active = true;
    v_branch_[g].tokens.resize(metadata_.ngram);
    v_branch_[g].i_batch.resize(metadata_.ngram);
    v_branch_[g].seq_id = metadata_.window + 1 + g;
    v_branch_[g].i_batch[0] = 0;
    v_branch_[g].tokens[0] = cur_token;
  }

  for (int j = 0; j < metadata_.ngram - 1; j++) {
    for (int g = 0; g < g_cur; g++) {
      const int idx = cur_token * (metadata_.ngram - 1) * metadata_.gcap +
          g * (metadata_.ngram - 1);
      const int32_t t = ngrams_pool_.tokens[idx + j];
      v_branch_[g].tokens[j + 1] = t;
      v_branch_[g].i_batch[j + 1] = j + 1;
    }
  }
}

template <typename T>
void MultimodalLhdTokenGenerator<T>::update_ngrams_pool() {
  std::vector<int32_t> ngram(metadata_.ngram - 1);
  // n-gram pool generation
  for (int f = 0; f < metadata_.window; ++f) {
    const int ft = lhd_branch_prev_[f]; // first token of the n-gram

    for (int j = 0; j < metadata_.ngram - 1; ++j) {
      ngram[j] = lhd_branch_[j][f];
    }

    // filter-out repeating n-grams
    {
      bool is_unique = true;
      for (int k = 0; k < ngrams_pool_.cnt[ft]; ++k) {
        // calculate the related idx by the first n-gram token
        const int idx = ft * (metadata_.ngram - 1) * metadata_.gcap +
            k * (metadata_.ngram - 1);

        bool is_match = true;
        for (int j = 0; j < metadata_.ngram - 1; ++j) {
          if (ngrams_pool_.tokens[idx + j] != ngram[j]) {
            is_match = false;
            break;
          }
        }

        // if n-gram match all, discard one of them
        if (is_match) {
          is_unique = false;
          break;
        }
      }
      if (!is_unique) {
        continue;
      }
    }

    const int head = ngrams_pool_.head[ft];
    const int idx = ft * (metadata_.ngram - 1) * metadata_.gcap +
        head * (metadata_.ngram - 1);

    for (int i = 0; i < metadata_.ngram - 1; i++) {
      // update the n-gram pool with new n-gram
      ngrams_pool_.tokens[idx + i] = ngram[i];
    }

    ngrams_pool_.cnt[ft] =
        std::min(metadata_.gcap, (int32_t)ngrams_pool_.cnt[ft] + 1);
    ngrams_pool_.head[ft] = (head + 1) % metadata_.gcap;
    ngrams_pool_.n_total++;
  }
}

template <typename T>
void MultimodalLhdTokenGenerator<T>::update_lookahead_branch(
    const executorch::aten::Tensor& logits_tensor) {
  for (int i = 0; i < metadata_.window; i++) {
    lhd_branch_prev_[i] = lhd_branch_[0][i];
  }

  for (int j = 0; j < metadata_.ngram - 2; j++) {
    lhd_branch_[j] = lhd_branch_[j + 1];
  }

  // sample from the last level
  for (int i = 0; i < metadata_.window; i++) {
    size_t sample_idx = (metadata_.ngram - 2) * metadata_.window + i;
    lhd_branch_[metadata_.ngram - 2][i] =
        this->decoder_runner_->logits_to_token(logits_tensor, sample_idx);
  }
}

template <typename T>
Result<int64_t> MultimodalLhdTokenGenerator<T>::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(
      !tokens.empty(), "Token generation loop shouldn't take empty tokens");
  // position in the sequence
  int64_t pos = start_pos;
  int64_t prev_pos;
  // number of match tokens
  int32_t n_accept{0};
  std::vector<uint64_t> result_tokens;
  uint64_t cur_token = tokens.back();
  uint64_t prev_token;
  result_tokens.push_back(cur_token);

  // Manage the inputs of lookahead decoding
  std::vector<int32_t> input_pos;
  std::vector<uint64_t> input_tokens;
  input_tokens.reserve(metadata_.ar_len);
  input_pos.reserve(metadata_.ar_len);

  // Rearrange KV cache first and initialize the input and output of KV cache
  this->kv_manager_->rearrange_cache(metadata_.ar_len);

  // Initialize attention mask with pos
  init_attention_mask(pos);

  // Initialize Lookahead branch at first generation
  if (!is_lhd_branch_initialized_) {
    ET_LOG(Info, "Initialize Lookahead branch");
    init_lookahead_branch(tokens);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      this->decoder_runner_->set_outputs(
          this->method_name_, this->output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      this->method_name_.c_str());

  // Generate tokens
  while (pos < seq_len - 1) {
    std::vector<bool> selected(metadata_.ar_len, false);

    input_tokens.clear();
    input_pos.clear();

    // fill the first token of the first level
    input_tokens.push_back(cur_token);
    input_pos.push_back(pos);

    // fill the remaining WINDOW - 1 tokens for the first level
    for (int i = 1; i < metadata_.window; ++i) {
      input_tokens.push_back(lhd_branch_[0][i]);
      input_pos.push_back(pos + i);
    }

    // fill the rest of the levels
    for (int i = 1; i < metadata_.ngram - 1; ++i) {
      for (int j = 0; j < metadata_.window; ++j) {
        input_tokens.push_back(lhd_branch_[i][j]);
        input_pos.push_back(pos + i + j);
      }
    }
    // Verification Branch Init
    init_verification_branch(cur_token);

    for (int g = 0; g < v_branch_.size(); g++) {
      for (int j = 0; j < metadata_.ngram - 1; j++) {
        input_tokens.push_back(v_branch_[g].tokens[j + 1]);
        input_pos.push_back(pos + j + 1);
      }
    }

    prepare_io(input_tokens, input_pos);

    // Run inference
    auto logits_res =
        this->decoder_runner_->step(this->method_name_, this->inputs_);
    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    executorch::aten::Tensor& logits_tensor = logits_res.get();
    prev_pos = pos;

    // verification branch seq-id
    size_t seq_id_best = 0;
    // max hit pos
    size_t i_batch_best = 0;

    // Lookahead decoding and verification
    for (int v = 0; v < metadata_.ngram; ++v) {
      // Verification
      int i_batch = 0;
      if (v > 0) {
        for (int g = 0; g < v_branch_.size(); g++) {
          // record the best matched seq and pos
          if (v_branch_[g].active) {
            i_batch = v_branch_[g].i_batch[v];
            i_batch_best = i_batch;
            seq_id_best = v_branch_[g].seq_id;
            ++n_accept;
            break;
          }
        }
        if (i_batch == 0) {
          break;
        }
      }

      size_t sample_idx;
      if (seq_id_best == 0)
        sample_idx = 0;
      else
        sample_idx = metadata_.window * (metadata_.ngram - 1) +
            (seq_id_best - (metadata_.window + 1)) * (metadata_.ngram - 1) +
            i_batch - 1;

      // vector selected set
      selected[sample_idx] = true;

      prev_token = cur_token;
      // sampler from logits all
      this->stats_->on_sampling_begin();
      cur_token =
          this->decoder_runner_->logits_to_token(logits_tensor, sample_idx);
      this->stats_->on_sampling_end();
      result_tokens.push_back(cur_token);
      pos++;

      // print the token as string, decode it with the Tokenizer object
      token_callback(
          ET_UNWRAP_TOKENIZER(this->tokenizer_->decode(prev_token, cur_token)));

      // data-dependent terminating condition: we have n_eos_ number of EOS
      if (this->eos_ids_->count(cur_token) > 0) {
        printf("\n");
        ET_LOG(Info, "\nReached to the end of generation");
        break;
      }

      // if verify pass, check the next sample token until verifying failed
      for (int g = 0; g < v_branch_.size(); g++) {
        // update the n-gram active status
        if (v_branch_[g].active) {
          if (v == metadata_.ngram - 1) {
            v_branch_[g].active = false;
          } else {
            if (cur_token != v_branch_[g].tokens[v + 1]) {
              v_branch_[g].active = false;
            }
          }
        }
      }

      // only update n-grams pools and lookahead branch when v=0
      if (v == 0) {
        // update lookahead branch
        update_lookahead_branch(logits_tensor);
        // update n-grams pool
        update_ngrams_pool();
      }
    } // end of verify loop

    if (pos > metadata_.context_len - metadata_.ar_len) {
      printf("\n");
      ET_LOG(Info, "\nReached to the maximum sequence length");
      break;
    }
    // Update KV Cache with the output results
    int32_t n_update = pos - prev_pos;
    this->kv_manager_->update_cache(
        metadata_.ar_len, prev_pos, n_update, selected);

    // Update attention mask with current position
    this->kv_manager_->update_attention_mask(
        this->attention_mask_.data, metadata_.ar_len, prev_pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      this->kv_manager_->update_attention_mask(
          this->window_attention_mask_.data,
          metadata_.ar_len,
          prev_pos,
          n_update,
          metadata_.sliding_window,
          position_offset_);
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (this->eos_ids_->count(cur_token) > 0) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  ET_LOG(
      Info,
      "Lookahead Decoding: n_generated = %ld / n_accept = %d",
      pos - start_pos,
      n_accept);

  return pos - start_pos;
}

// Explicit instantiations
template class MultimodalLhdTokenGenerator<uint16_t>;
template class MultimodalLhdTokenGenerator<uint8_t>;

} // namespace example
