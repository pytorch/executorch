/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

namespace vkcompute {

class LwgShape final {
 private:
  utils::uvec3 axis_weights_;

 public:
  explicit LwgShape();
  explicit LwgShape(uint32_t x, uint32_t y, uint32_t z);
  explicit LwgShape(const utils::uvec3& axis_weights);

  uint32_t operator[](int idx) const;

  bool is_valid() const;
  uint32_t allocate_exponents(
      uint32_t num_exponents,
      const utils::uvec3& exponent_limits,
      utils::uvec3& exponents) const;
  utils::uvec3 distribute_exponents(uint32_t target_exponent) const;
};

extern const LwgShape kLinearLwg;
extern const LwgShape kSquareLwg;
extern const LwgShape kCubeLwg;

enum class DispatchGridIntent : uint8_t {
  // The caller defines how invocation coordinates map to logical work.
  Explicit,
  // Invocation coordinates address a flattened one-dimensional workload.
  Linear,
  // Invocation coordinates directly match tensor texture extents.
  TextureExtents,
  // Invocation coordinates address operator-defined output tiles.
  Tiled,
};

constexpr DispatchGridIntent kExplicitWorkGrid = DispatchGridIntent::Explicit;
constexpr DispatchGridIntent kLinearWorkGrid = DispatchGridIntent::Linear;
constexpr DispatchGridIntent kTextureExtentsWorkGrid =
    DispatchGridIntent::TextureExtents;
constexpr DispatchGridIntent kTiledWorkGrid = DispatchGridIntent::Tiled;

class GlobalWorkGrid;

class LocalWorkGroup final {
 private:
  // These store the exponents e of their corresponding power-of-two values.
  uint32_t target_total_nthreads_exp_;
  utils::uvec3 xyz_exponents_;
  LwgShape target_lwg_shape_;

 public:
  explicit LocalWorkGroup();
  explicit LocalWorkGroup(
      uint32_t x,
      uint32_t y,
      uint32_t z,
      uint32_t target_total_nthreads = 64u);
  explicit LocalWorkGroup(
      const utils::uvec3& vec,
      uint32_t target_total_nthreads = 64u);
  explicit LocalWorkGroup(
      const LwgShape& target_lwg_shape,
      uint32_t target_total_nthreads = 64u);

  explicit operator utils::uvec3() const;
  uint32_t operator[](int idx) const;
  bool operator==(const LocalWorkGroup& other) const;
  bool operator!=(const LocalWorkGroup& other) const;

  uint32_t x() const;
  uint32_t y() const;
  uint32_t z() const;
  uint32_t target_total_nthreads() const;

  bool is_valid() const;
  uint32_t nthreads() const;
  void validate(const utils::uvec3& max_lwg, uint32_t max_nthreads) const;
  void fit_to_global(const GlobalWorkGrid& gwg);
};

class GlobalWorkGrid final {
 private:
  utils::uvec3 extents_;
  DispatchGridIntent intent_;
  LocalWorkGroup required_lwg_;

 public:
  GlobalWorkGrid(const utils::uvec3& extents, DispatchGridIntent intent);
  GlobalWorkGrid(
      const utils::uvec3& extents,
      DispatchGridIntent intent,
      const LocalWorkGroup& required_lwg);

  bool operator==(const GlobalWorkGrid& other) const;
  bool operator!=(const GlobalWorkGrid& other) const;

  const utils::uvec3& extents() const;
  const LocalWorkGroup& required_lwg_size() const;
  DispatchGridIntent intent() const;
  bool is_linear() const;

  void wrap_linear_dispatch(
      const utils::uvec3& max_wg_count,
      uint32_t target_total_nthreads = 64u);
};

} // namespace vkcompute
