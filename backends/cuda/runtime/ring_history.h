/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::backends::cuda {

// Host undo history for a device ring with no per-owner write headroom.
// A completed step arrives as [planes, step_tokens, row_bytes]; only real
// segments enter the archive. Page tags distinguish retained logical rows
// from stale rows after wrapping. Clones share pages until a later write.
// Not thread-safe: the owning cache serializes calls and fences the incoming
// device-to-host transfer before append(). Returned rows remain valid until
// the next mutation, so a restore must finish before pages can be released.
class RingHistory final {
 public:
  struct Run {
    int32_t owner;
    int packed_start;
    int length;
    int absolute_start;
  };

  struct Rows {
    const uint8_t* data;
    int count;
  };

  static runtime::Result<std::unique_ptr<RingHistory>> create(
      int window,
      int max_write,
      int context_limit,
      size_t row_bytes,
      size_t planes) {
    if (window <= 0 || max_write <= 0 || context_limit < max_write ||
        row_bytes == 0 || planes == 0 ||
        row_bytes > std::numeric_limits<size_t>::max() / kPageRows / planes ||
        row_bytes > std::numeric_limits<size_t>::max() / max_write / planes) {
      return runtime::Error::InvalidArgument;
    }
    const int rows = static_cast<int>(std::min<int64_t>(
        context_limit, static_cast<int64_t>(window) + max_write - 1));
    return std::unique_ptr<RingHistory>(new RingHistory(
        std::min(window, context_limit),
        max_write,
        context_limit,
        rows,
        row_bytes,
        planes));
  }

  runtime::Error append(
      runtime::Span<const Run> runs,
      runtime::Span<const uint8_t> packed,
      int step_tokens) {
    if (runs.empty() || step_tokens <= 0 || step_tokens > max_write_ ||
        packed.size() !=
            planes_ * static_cast<size_t>(step_tokens) * row_bytes_) {
      return runtime::Error::InvalidArgument;
    }
    // Validate the whole step before changing any owner or shared page.
    int packed_end = 0;
    std::map<int32_t, int> ends;
    for (const auto& run : runs) {
      if (run.owner < 0 || run.packed_start < packed_end || run.length <= 0 ||
          run.length > step_tokens ||
          run.packed_start > step_tokens - run.length ||
          run.absolute_start < 0 ||
          run.absolute_start > context_limit_ - run.length) {
        return runtime::Error::InvalidArgument;
      }
      const auto existing = owners_.find(run.owner);
      auto end = ends.emplace(
                         run.owner,
                         existing == owners_.end() ? 0 : existing->second.end)
                     .first;
      if (run.absolute_start != end->second) {
        return runtime::Error::InvalidState;
      }
      end->second += run.length;
      packed_end = run.packed_start + run.length;
    }
    for (const auto& run : runs) {
      auto& owner = owners_[run.owner];
      if (owner.pages.empty()) {
        owner.pages.resize((rows_ - 1) / kPageRows + 1);
      }
      int done = 0;
      while (done < run.length) {
        const int absolute = run.absolute_start + done;
        const int physical = absolute % rows_;
        const int page_row = physical % kPageRows;
        const int count = std::min(
            run.length - done,
            std::min(rows_ - physical, kPageRows - page_row));
        auto& page = owner.pages[physical / kPageRows];
        if (!page) {
          page = std::make_shared<Page>(planes_ * kPageRows * row_bytes_);
        } else if (page.use_count() > 1) {
          page = std::make_shared<Page>(*page);
        }
        for (size_t plane = 0; plane < planes_; ++plane) {
          std::memcpy(
              page->bytes.data() + (plane * kPageRows + page_row) * row_bytes_,
              packed.data() +
                  (plane * step_tokens + run.packed_start + done) * row_bytes_,
              static_cast<size_t>(count) * row_bytes_);
        }
        for (int row = 0; row < count; ++row) {
          page->positions[page_row + row] = absolute + row;
        }
        done += count;
      }
      owner.end = run.absolute_start + run.length;
      owner.written = std::max(owner.written, owner.end);
    }
    return runtime::Error::Ok;
  }

  std::optional<Rows>
  read(int32_t owner_id, int start, int max_rows, size_t plane) const {
    const auto owner = owners_.find(owner_id);
    if (owner == owners_.end() || start < 0 || start >= owner->second.end ||
        max_rows <= 0 || plane >= planes_) {
      return std::nullopt;
    }
    const int physical = start % rows_;
    const int page_row = physical % kPageRows;
    const auto& page = owner->second.pages[physical / kPageRows];
    if (!page) {
      return std::nullopt;
    }
    const int count = std::min(
        std::min(max_rows, owner->second.end - start),
        std::min(rows_ - physical, kPageRows - page_row));
    for (int row = 0; row < count; ++row) {
      if (page->positions[page_row + row] != start + row) {
        return std::nullopt;
      }
    }
    return Rows{
        page->bytes.data() + (plane * kPageRows + page_row) * row_bytes_,
        count};
  }

  bool clone(int32_t source, int32_t destination, int keep) {
    if (destination < 0 || owners_.count(destination) != 0 ||
        !has_window(source, keep)) {
      return false;
    }
    const auto& original = owners_.at(source);
    owners_.emplace(destination, Owner{original.pages, keep, original.written});
    return true;
  }

  bool rewind(int32_t owner, int keep) {
    if (!has_window(owner, keep)) {
      return false;
    }
    owners_.at(owner).end = keep;
    return true;
  }

  void erase(int32_t owner) {
    owners_.erase(owner);
  }

  void clear() {
    owners_.clear();
  }

  bool has_window(int32_t owner, int keep) const {
    const auto found = owners_.find(owner);
    if (found == owners_.end() ||
        keep < std::max(0, found->second.written - max_write_) ||
        keep > found->second.end) {
      return false;
    }
    for (int start = std::max(0, keep - window_ + 1); start < keep;) {
      const auto chunk = read(owner, start, keep - start, 0);
      if (!chunk) {
        return false;
      }
      start += chunk->count;
    }
    return true;
  }

  // Unique payload and row-tag bytes, excluding container overhead.
  size_t resident_bytes() const {
    std::unordered_set<const Page*> pages;
    size_t bytes = 0;
    for (const auto& [id, owner] : owners_) {
      for (const auto& page : owner.pages) {
        if (page && pages.insert(page.get()).second) {
          bytes += page->bytes.size() + sizeof(page->positions);
        }
      }
    }
    return bytes;
  }

 private:
  static constexpr int kPageRows = 64;

  struct Page {
    explicit Page(size_t size) : bytes(size) {
      positions.fill(-1);
    }
    std::vector<uint8_t> bytes;
    std::array<int, kPageRows> positions;
  };

  struct Owner {
    std::vector<std::shared_ptr<Page>> pages;
    int end{0};
    int written{0};
  };

  RingHistory(
      int window,
      int max_write,
      int context_limit,
      int rows,
      size_t row_bytes,
      size_t planes)
      : window_(window),
        max_write_(max_write),
        context_limit_(context_limit),
        rows_(rows),
        row_bytes_(row_bytes),
        planes_(planes) {}

  const int window_;
  const int max_write_;
  const int context_limit_;
  const int rows_;
  const size_t row_bytes_;
  const size_t planes_;
  std::map<int32_t, Owner> owners_;
};

} // namespace executorch::backends::cuda
