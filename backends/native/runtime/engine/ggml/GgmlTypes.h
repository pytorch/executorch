// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <ggml.h>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>

namespace ptn::ggml_detail {

size_t checked_numel(const std::vector<int64_t>& sizes);

size_t checked_nbytes(const std::vector<int64_t>& sizes, ScalarType dtype);

const PackedQuant* packed_quant(const TensorMeta& meta);

size_t packed_nbytes(const TensorMeta& meta);

const AffineGroupQuant* affine_quant(const TensorMeta& meta);

const AffineGroupQuant& q4_affine(const TensorMeta& meta);

std::vector<float> to_float_vector(ByteSpan bytes, ScalarType dtype);

std::vector<uint8_t> affine_int4_to_q4_0(
    const TensorMeta& meta,
    ByteSpan packed,
    const std::vector<float>& scales);

ggml_type compute_type(const TensorMeta& meta);

std::array<int64_t, GGML_MAX_DIMS> reversed_extents(
    const std::vector<int64_t>& sizes);

ggml_tensor* new_tensor(
    ggml_context* ctx,
    ggml_type type,
    const std::vector<int64_t>& sizes);

std::vector<uint8_t> narrow_to_int32(ByteSpan bytes);

} // namespace ptn::ggml_detail
