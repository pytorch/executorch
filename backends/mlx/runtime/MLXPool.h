/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "MLXExecutor.h" // Tensor, StreamOrDevice

namespace executorch {
namespace backends {
namespace mlx {

// Per-layer K or V store, SDPA-major [1, H, slots, D] (cells on axis 2). The
// caller hands down physical slots -- a contiguous range, or one index per
// token -- having already applied any ring modulo, so the pool is
// layout-agnostic: layouts differ only in how many slots the layer asks for and
// where in them a step lands.
class Pool {
 public:
  // initial_slots above max_slots is clamped, not rejected: the config default
  // exceeds the cap of any smaller cache, so this is the normal path.
  Pool(int initial_slots, int max_slots, int H, int D, ::mlx::core::Dtype dtype)
      : dtype_(dtype),
        max_slots_(max_slots),
        buf_(::mlx::core::zeros(
            ::mlx::core::Shape{1, H, std::min(initial_slots, max_slots), D},
            dtype)) {}

  // Place `update` at slot `start`, casting to the storage dtype if it differs.
  void write(int start, int len, const Tensor& update, StreamOrDevice s) {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    if (start < 0 || start + len > max_slots_) {
      throw std::runtime_error("Pool::write: run out of bounds");
    }
    if (static_cast<int>(update.shape(2)) != len) {
      throw std::runtime_error("Pool::write: update length != run length");
    }
    if (static_cast<int>(update.shape(1)) != H ||
        static_cast<int>(update.shape(3)) != D) {
      throw std::runtime_error("Pool::write: K/V heads/dim mismatch");
    }
    maybe_grow(start + len, s);
    const Tensor u = update.dtype() == dtype_
        ? update
        : ::mlx::core::astype(update, dtype_, s);
    buf_ = ::mlx::core::slice_update(
        buf_,
        u,
        ::mlx::core::Shape{0, 0, start, 0},
        ::mlx::core::Shape{1, H, start + len, D},
        s);
  }

  // Place `update` one token per slot, at `cells[i]` for token i. The cells
  // must be distinct. They refill from below as sequences are removed, so
  // a step's slots need not be contiguous or ordered.
  void write_cells(
      const std::vector<int32_t>& cells,
      const Tensor& update,
      StreamOrDevice s) {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    const int T = static_cast<int>(cells.size());
    if (static_cast<int>(update.shape(2)) != T) {
      throw std::runtime_error(
          "Pool::write_cells: update length != cell count");
    }
    if (static_cast<int>(update.shape(1)) != H ||
        static_cast<int>(update.shape(3)) != D) {
      throw std::runtime_error("Pool::write_cells: K/V heads/dim mismatch");
    }
    int high = 0;
    for (int32_t cell : cells) {
      if (cell < 0 || cell >= max_slots_) {
        throw std::runtime_error("Pool::write_cells: cell out of bounds");
      }
      high = std::max(high, cell + 1);
    }
    maybe_grow(high, s);
    // put_along_axis broadcasts the one index per token across heads and
    // head_dim, and casts the update to the storage dtype.
    buf_ = ::mlx::core::put_along_axis(
        buf_,
        ::mlx::core::array(
            cells.data(), ::mlx::core::Shape{1, 1, T, 1}, ::mlx::core::int32),
        update,
        2,
        s);
  }

  // Slots [start, start+len). A ring read starts mid-pool, so the start matters
  // here as much as it does for a write.
  Tensor read(int start, int len, StreamOrDevice s) const {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    if (start < 0 || start + len > slots()) {
      throw std::runtime_error("Pool::read: run out of bounds");
    }
    return ::mlx::core::slice(
        buf_,
        ::mlx::core::Shape{0, 0, start, 0},
        ::mlx::core::Shape{1, H, start + len, D},
        ::mlx::core::Shape{1, 1, 1, 1},
        s);
  }

  // Slots currently allocated; grows toward max_slots on demand.
  int slots() const {
    return static_cast<int>(buf_.shape(2));
  }

 private:
  // Make room for `needed` slots, growing only if the pool is short: double
  // until it fits, never past max_slots_. Cells keep their index, so growth is
  // a zero-pad on the cell axis.
  void maybe_grow(int needed, StreamOrDevice s) {
    const int cur = slots();
    if (needed <= cur) {
      return;
    }
    int next = std::max(cur, 1); // an empty pool has nothing to double
    while (next < needed) {
      next *= 2;
    }
    // The last doubling can overshoot; write() already bounds `needed` by
    // max_slots_, so clamping here cannot undershoot it.
    next = std::min(next, max_slots_);
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    Tensor pad =
        ::mlx::core::zeros(::mlx::core::Shape{1, H, next - cur, D}, dtype_);
    buf_ = ::mlx::core::concatenate(std::vector<Tensor>{buf_, pad}, 2, s);
  }

  ::mlx::core::Dtype dtype_;
  int max_slots_;
  Tensor buf_;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
