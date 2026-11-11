/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// BatchControl over a cache that holds one sequence, so a SessionTable serves
// one session from it: an executor over a single-sequence cache. Sequence 0 is
// the only id, live from seq_new until seq_rm, which clears it; clone is
// refused, since the cache has nowhere to put a second history.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {
namespace util {

class ET_EXPERIMENTAL SequenceBatchControl final
    : public ::executorch::extension::llm::cache::BatchControl {
 public:
  // `sequence` must outlive this.
  explicit SequenceBatchControl(
      ::executorch::extension::llm::cache::SequenceControl& sequence)
      : sequence_(sequence) {}

  int capacity() const override {
    return sequence_.capacity();
  }
  void clear() override {
    sequence_.clear();
  }
  bool declare_step(const std::vector<std::int32_t>& seq_ids) override {
    return live_ &&
        std::all_of(
               seq_ids.begin(),
               seq_ids.end(),
               [](std::int32_t id) { return id == 0; }) &&
        sequence_.can_extend(static_cast<int>(seq_ids.size()));
  }
  std::optional<int> max_seqs() const override {
    return 1;
  }
  std::optional<std::int32_t> seq_new() override {
    if (live_) {
      return std::nullopt;
    }
    sequence_.clear();
    live_ = true;
    return 0;
  }
  std::optional<std::int32_t> seq_clone(
      std::int32_t /*src*/,
      std::optional<int> /*upto*/) override {
    return std::nullopt;
  }
  bool seq_rm(std::int32_t seq_id) override {
    if (!holds(seq_id)) {
      return false;
    }
    sequence_.clear();
    live_ = false;
    return true;
  }
  bool rewind(std::int32_t seq_id, int position) override {
    return holds(seq_id) && sequence_.rewind(position);
  }
  int pos(std::int32_t seq_id) const override {
    return holds(seq_id) ? sequence_.length() : 0;
  }

 private:
  bool holds(std::int32_t seq_id) const {
    return live_ && seq_id == 0;
  }

  ::executorch::extension::llm::cache::SequenceControl& sequence_;
  bool live_ = false;
};

} // namespace util
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
