// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace ptn {

size_t greedy_argmax_last_row(
    const std::vector<float>& logits,
    size_t row_size);
std::vector<size_t> top_logit_indices(const float* row, size_t n, size_t k);
std::unordered_set<uint64_t> parse_token_ids(std::string_view text);
size_t resolve_logits_output(
    size_t output_count,
    const std::vector<bool>& user_outputs);
// Sequence length of the KV cache that operator `target` writes through its
// `input` argument, given that argument's upper-bound sizes, or 0 if the
// argument is not a cache. llama.update_cache writes [batch, sequence, heads,
// head_dim] caches, and HuggingFace static caches take index_put_ rows along
// dim -2 of [batch, heads, sequence, head_dim]. Writes into anything that is
// not a rank-4 tensor with a positive sequence length are not caches.
size_t kv_cache_sequence_length(
    std::string_view target,
    std::string_view input,
    const std::vector<int64_t>& sizes);
// Tokens to generate after the prompt; throws if the prompt does not fit. A
// cache_length of 0, for a method without a recognizable cache, falls back to
// the token input's upper bound.
size_t generation_limit(
    size_t prompt_length,
    size_t max_new_tokens,
    size_t max_seq_length,
    size_t cache_length,
    size_t token_upper_bound);

struct Closeness {
  double max_abs_error = 0.0;
  size_t nan_count = 0;
  bool close = true;
};
// Largest non-NaN |got - want|, the number of elements whose difference is NaN,
// and whether every element is within atol + rtol * |want|. Equal values,
// infinities included, are close; a NaN never is.
Closeness compare_close(
    const float* got,
    const float* want,
    size_t n,
    double atol,
    double rtol);

} // namespace ptn
