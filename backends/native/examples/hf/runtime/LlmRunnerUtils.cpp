// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/hf/runtime/LlmRunnerUtils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ptn {
namespace {

// Same index as std::max_element: the first maximum, where NaN never compares
// greater. The branch-free max pass vectorizes, unlike max_element's loop.
size_t first_argmax(const float* row, size_t n) {
  constexpr size_t kLanes = 16;
  float lanes[kLanes];
  std::fill(lanes, lanes + kLanes, row[0]);
  size_t i = 0;
  for (; i + kLanes <= n; i += kLanes) {
    for (size_t j = 0; j < kLanes; ++j) {
      lanes[j] = row[i + j] > lanes[j] ? row[i + j] : lanes[j];
    }
  }
  // Every lane starts at row[0] and only grows, so the largest lane is the
  // running maximum (or row[0] itself when it is NaN).
  float best = *std::max_element(lanes, lanes + kLanes);
  for (; i < n; ++i) {
    best = row[i] > best ? row[i] : best;
  }
  for (size_t base = 0; base < n; base += kLanes) {
    const size_t end = std::min(n, base + kLanes);
    bool found = false;
    for (size_t k = base; k < end; ++k) {
      found |= row[k] == best;
    }
    if (found) {
      return static_cast<size_t>(std::find(row + base, row + end, best) - row);
    }
  }
  // Only a leading NaN leaves no element equal to best.
  return static_cast<size_t>(std::max_element(row, row + n) - row);
}

} // namespace

// cppcheck-suppress unusedFunction
size_t greedy_argmax_last_row(
    const std::vector<float>& logits,
    size_t row_size) {
  if (row_size == 0 || logits.size() < row_size ||
      logits.size() % row_size != 0) {
    throw std::invalid_argument("logits do not contain complete rows");
  }
  return first_argmax(logits.data() + logits.size() - row_size, row_size);
}

// cppcheck-suppress unusedFunction
std::vector<size_t> top_logit_indices(const float* row, size_t n, size_t k) {
  std::vector<size_t> ranks(n);
  std::iota(ranks.begin(), ranks.end(), 0);
  const size_t count = std::min(k, n);
  // NaNs rank first; a bare > is not a strict weak ordering with NaNs.
  std::partial_sort(
      ranks.begin(),
      ranks.begin() + count,
      ranks.end(),
      [row](size_t a, size_t b) {
        return row[a] > row[b] || (std::isnan(row[a]) && !std::isnan(row[b]));
      });
  ranks.resize(count);
  return ranks;
}

// cppcheck-suppress unusedFunction
std::unordered_set<uint64_t> parse_token_ids(std::string_view text) {
  if (text.empty()) {
    throw std::invalid_argument("token IDs must not be empty");
  }
  std::unordered_set<uint64_t> ids;
  size_t offset = 0;
  while (true) {
    const size_t comma = text.find(',', offset);
    const std::string item(text.substr(offset, comma - offset));
    // std::stoull accepts a leading sign and wraps negative values.
    if (item.empty() || !std::isdigit(static_cast<unsigned char>(item[0]))) {
      throw std::invalid_argument("invalid token ID: " + item);
    }
    size_t parsed = 0;
    const uint64_t id = std::stoull(item, &parsed);
    if (parsed != item.size()) {
      throw std::invalid_argument("invalid token ID: " + item);
    }
    ids.insert(id);
    if (comma == std::string_view::npos) {
      break;
    }
    offset = comma + 1;
  }
  return ids;
}

// cppcheck-suppress unusedFunction
size_t resolve_logits_output(
    size_t output_count,
    const std::vector<bool>& user_outputs) {
  if (user_outputs.empty()) {
    if (output_count == 1) {
      return 0;
    }
    throw std::invalid_argument("method has no unique logits output");
  }
  if (user_outputs.size() != output_count) {
    throw std::invalid_argument("output metadata does not match graph outputs");
  }
  const auto user_output_count =
      std::count(user_outputs.begin(), user_outputs.end(), true);
  if (user_output_count > 1) {
    throw std::invalid_argument("method has multiple user outputs");
  }
  if (user_output_count == 0) {
    throw std::invalid_argument("method has no unique logits output");
  }
  // The logits output is the only user output, so its public index is 0.
  return 0;
}

// cppcheck-suppress unusedFunction
size_t kv_cache_sequence_length(
    std::string_view target,
    std::string_view input,
    const std::vector<int64_t>& sizes) {
  std::string_view cache_name;
  size_t sequence_dim = 0;
  if (target == "torch.ops.llama.update_cache.default") {
    cache_name = "cache";
    sequence_dim = 1;
  } else if (target == "torch.ops.aten.index_put_.default") {
    cache_name = "self";
    sequence_dim = 2;
  } else {
    return 0;
  }
  if (input != cache_name || sizes.size() != 4 || sizes[sequence_dim] <= 0) {
    return 0;
  }
  return static_cast<size_t>(sizes[sequence_dim]);
}

// cppcheck-suppress unusedFunction
size_t generation_limit(
    size_t prompt_length,
    size_t max_new_tokens,
    size_t max_seq_length,
    size_t cache_length,
    size_t token_upper_bound) {
  // The exporters guarantee the cache holds the token input's upper bound. The
  // last generated token is never fed back, so one more token than the cache
  // holds fits.
  const size_t capacity = cache_length > 0 ? cache_length : token_upper_bound;
  const size_t max_total_tokens = std::min(max_seq_length, capacity + 1);
  if (prompt_length >= max_total_tokens) {
    throw std::runtime_error(
        "prompt of " + std::to_string(prompt_length) +
        " tokens does not fit --max_seq_length or the model's " +
        std::to_string(capacity) + "-token cache");
  }
  return std::min(max_new_tokens, max_total_tokens - prompt_length);
}

} // namespace ptn
