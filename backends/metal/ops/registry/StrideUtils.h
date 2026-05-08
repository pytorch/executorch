/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Stride helpers — broadcast resolution + dim collapsing for general
// elementwise paths. Mirrors mlx/backend/common/utils.

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

using exec_aten::ArrayRef;
using exec_aten::SizesType;
using runtime::etensor::Tensor;

// Compute strides for `shape` aligned to `out_shape` such that broadcast
// dimensions (where the input has size 1 or is missing) get stride 0,
// and non-broadcast dimensions take the input's actual stride.
inline std::vector<int64_t> broadcastStrides(
    ArrayRef<SizesType> in_shape,
    ArrayRef<exec_aten::StridesType> in_strides,
    const std::vector<SizesType>& out_shape) {
  std::vector<int64_t> strides(out_shape.size(), 0);
  int offset =
      static_cast<int>(out_shape.size()) - static_cast<int>(in_shape.size());
  for (int i = static_cast<int>(in_shape.size()) - 1; i >= 0; --i) {
    if (in_shape[i] == out_shape[i + offset]) {
      strides[i + offset] = static_cast<int64_t>(in_strides[i]);
    }
    // else: input dim is 1 (broadcast) or missing -> stride stays 0
  }
  return strides;
}

// Merges adjacent dims that are contiguous in *every* input's stride layout.
// Reduces effective ndim for the General kernel.
inline std::pair<std::vector<SizesType>, std::vector<std::vector<int64_t>>>
collapseContiguousDims(
    const std::vector<SizesType>& shape,
    const std::vector<std::vector<int64_t>>& strides_per_input) {
  const size_t ndim = shape.size();
  const size_t nin = strides_per_input.size();

  std::vector<SizesType> out_shape;
  std::vector<std::vector<int64_t>> out_strides(nin);

  if (ndim == 0) {
    return {out_shape, out_strides};
  }

  out_shape.push_back(shape[ndim - 1]);
  for (size_t k = 0; k < nin; ++k) {
    out_strides[k].push_back(strides_per_input[k][ndim - 1]);
  }

  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    SizesType inner_size = out_shape.back();
    bool can_merge = (shape[i] == 1) || (inner_size == 1);
    if (!can_merge) {
      can_merge = true;
      for (size_t k = 0; k < nin; ++k) {
        int64_t outer = strides_per_input[k][i];
        int64_t inner = out_strides[k].back();
        bool both_zero = (outer == 0) && (inner == 0);
        bool packed = (outer == inner * inner_size);
        if (!both_zero && !packed) {
          can_merge = false;
          break;
        }
      }
    }
    if (can_merge) {
      out_shape.back() = shape[i] * inner_size;
      // Merge rule: when inner has size 1, take outer's stride (which may
      // be 0 for broadcast); else keep inner (the contiguous step or 0).
      for (size_t k = 0; k < nin; ++k) {
        int64_t outer = strides_per_input[k][i];
        if (inner_size == 1) {
          out_strides[k].back() = outer;
        }
      }
    } else {
      out_shape.push_back(shape[i]);
      for (size_t k = 0; k < nin; ++k) {
        out_strides[k].push_back(strides_per_input[k][i]);
      }
    }
  }

  std::reverse(out_shape.begin(), out_shape.end());
  for (auto& s : out_strides) {
    std::reverse(s.begin(), s.end());
  }
  return {out_shape, out_strides};
}

// Build packed row-major strides for `shape`. e.g. [2,3,4] -> [12,4,1].
inline std::vector<int64_t> makeContiguousStrides(
    const std::vector<SizesType>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<int64_t>(shape[i]);
  }
  return strides;
}

inline std::vector<int64_t> makeContiguousStrides(ArrayRef<SizesType> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<int64_t>(shape[i]);
  }
  return strides;
}

// True if `t` is column-major contiguous.
inline bool isColContiguous(const Tensor& t) {
  auto sizes = t.sizes();
  auto strides = t.strides();
  if (sizes.size() != strides.size()) return false;
  if (sizes.empty()) return true;
  int64_t expected = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 1) continue;
    if (strides[i] != expected) return false;
    expected *= sizes[i];
  }
  return true;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
