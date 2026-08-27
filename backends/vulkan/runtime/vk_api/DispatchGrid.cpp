/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/DispatchGrid.h>

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace vkcompute {

namespace {

constexpr uint32_t kInvalidExponent = std::numeric_limits<uint32_t>::max();

bool is_power_of_two(const uint32_t value) {
  return value > 0u && (value & (value - 1u)) == 0u;
}

uint32_t power_of_two_exponent(uint32_t value) {
  VK_CHECK_COND(is_power_of_two(value));

#if defined(__GNUC__) || defined(__clang__)
  return static_cast<uint32_t>(__builtin_ctz(value));
#elif defined(_MSC_VER)
  unsigned long exponent;
  _BitScanForward(&exponent, value);
  return static_cast<uint32_t>(exponent);
#else
  uint32_t exponent = 0u;
  while (value > 1u) {
    value >>= 1u;
    ++exponent;
  }
  return exponent;
#endif
}

uint32_t workgroup_size_exponent(const uint32_t value) {
  if (value == 0u) {
    return kInvalidExponent;
  }
  VK_CHECK_COND(
      is_power_of_two(value),
      "Local workgroup dimensions must be powers of two");
  return power_of_two_exponent(value);
}

uint32_t useful_exponent_limit(
    const uint32_t global_extent,
    const uint32_t target_exponent) {
  if (global_extent <= 1u) {
    return 0u;
  }

  const uint64_t useful_local_extent = uint64_t(global_extent) + 2u;
  uint32_t exponent = 0u;
  while (exponent < target_exponent &&
         (uint64_t(1u) << (exponent + 1u)) <= useful_local_extent) {
    ++exponent;
  }
  return exponent;
}

void allocate_by_exponent_capacity(
    uint32_t num_exponents,
    const utils::uvec3& exponent_limits,
    utils::uvec3& exponents) {
  while (num_exponents > 0u) {
    int32_t selected_axis = -1;
    uint32_t selected_capacity = 0u;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const uint32_t capacity = exponent_limits[axis] - exponents[axis];
      if (capacity > selected_capacity) {
        selected_axis = static_cast<int32_t>(axis);
        selected_capacity = capacity;
      }
    }

    if (selected_axis < 0) {
      break;
    }
    const uint32_t allocated = std::min(num_exponents, selected_capacity);
    exponents[static_cast<uint32_t>(selected_axis)] += allocated;
    num_exponents -= allocated;
  }
}

} // namespace

LwgShape::LwgShape() : axis_weights_{0u, 0u, 0u} {}

LwgShape::LwgShape(const uint32_t x, const uint32_t y, const uint32_t z)
    : axis_weights_{x, y, z} {}

LwgShape::LwgShape(const utils::uvec3& axis_weights)
    : axis_weights_(axis_weights) {}

uint32_t LwgShape::operator[](const int idx) const {
  return axis_weights_[idx];
}

bool LwgShape::is_valid() const {
  return axis_weights_[0] > 0u || axis_weights_[1] > 0u ||
      axis_weights_[2] > 0u;
}

uint32_t LwgShape::allocate_exponents(
    uint32_t num_exponents,
    const utils::uvec3& exponent_limits,
    utils::uvec3& exponents) const {
  // Use the D'Hondt highest-averages method to preserve the target ratio.
  while (num_exponents > 0u) {
    int32_t selected_axis = -1;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      if ((*this)[axis] == 0u || exponents[axis] >= exponent_limits[axis]) {
        continue;
      }
      if (selected_axis < 0) {
        selected_axis = static_cast<int32_t>(axis);
        continue;
      }

      const uint32_t selected = static_cast<uint32_t>(selected_axis);
      const uint64_t candidate_score =
          uint64_t((*this)[axis]) * (exponents[selected] + 1u);
      const uint64_t selected_score =
          uint64_t((*this)[selected]) * (exponents[axis] + 1u);
      if (candidate_score > selected_score) {
        selected_axis = static_cast<int32_t>(axis);
      }
    }

    if (selected_axis < 0) {
      break;
    }
    ++exponents[static_cast<uint32_t>(selected_axis)];
    --num_exponents;
  }
  return num_exponents;
}

utils::uvec3 LwgShape::distribute_exponents(
    const uint32_t target_exponent) const {
  VK_CHECK_COND(
      is_valid(), "Local workgroup shape must contain a nonzero component");

  utils::uvec3 exponents{0u, 0u, 0u};
  const utils::uvec3 exponent_limits{
      target_exponent, target_exponent, target_exponent};
  const uint32_t remaining =
      allocate_exponents(target_exponent, exponent_limits, exponents);
  VK_CHECK_COND(remaining == 0u);
  return exponents;
}

const LwgShape kLinearLwg{1u, 0u, 0u};
const LwgShape kSquareLwg{1u, 1u, 0u};
const LwgShape kCubeLwg{1u, 1u, 1u};

LocalWorkGroup::LocalWorkGroup()
    : target_total_nthreads_exp_(power_of_two_exponent(64u)),
      xyz_exponents_{kInvalidExponent, kInvalidExponent, kInvalidExponent},
      target_lwg_shape_() {}

LocalWorkGroup::LocalWorkGroup(
    const uint32_t x,
    const uint32_t y,
    const uint32_t z,
    const uint32_t target_total_nthreads)
    : target_total_nthreads_exp_(power_of_two_exponent(target_total_nthreads)),
      xyz_exponents_{
          workgroup_size_exponent(x),
          workgroup_size_exponent(y),
          workgroup_size_exponent(z)},
      target_lwg_shape_() {}

LocalWorkGroup::LocalWorkGroup(
    const utils::uvec3& vec,
    const uint32_t target_total_nthreads)
    : LocalWorkGroup(vec[0u], vec[1u], vec[2u], target_total_nthreads) {}

LocalWorkGroup::LocalWorkGroup(
    const LwgShape& target_lwg_shape,
    const uint32_t target_total_nthreads)
    : target_total_nthreads_exp_(power_of_two_exponent(target_total_nthreads)),
      xyz_exponents_{0u, 0u, 0u},
      target_lwg_shape_(target_lwg_shape) {
  xyz_exponents_ =
      target_lwg_shape_.distribute_exponents(target_total_nthreads_exp_);
}

LocalWorkGroup::operator utils::uvec3() const {
  return {x(), y(), z()};
}

uint32_t LocalWorkGroup::operator[](const int idx) const {
  const uint32_t exponent = xyz_exponents_[idx];
  return exponent == kInvalidExponent ? 0u : 1u << exponent;
}

bool LocalWorkGroup::operator==(const LocalWorkGroup& other) const {
  return xyz_exponents_ == other.xyz_exponents_;
}

bool LocalWorkGroup::operator!=(const LocalWorkGroup& other) const {
  return !(*this == other);
}

uint32_t LocalWorkGroup::x() const {
  return (*this)[0];
}

uint32_t LocalWorkGroup::y() const {
  return (*this)[1];
}

uint32_t LocalWorkGroup::z() const {
  return (*this)[2];
}

uint32_t LocalWorkGroup::target_total_nthreads() const {
  return 1u << target_total_nthreads_exp_;
}

bool LocalWorkGroup::is_valid() const {
  return x() > 0u && y() > 0u && z() > 0u;
}

uint32_t LocalWorkGroup::nthreads() const {
  return x() * y() * z();
}

void LocalWorkGroup::validate(
    const utils::uvec3& max_lwg,
    const uint32_t max_nthreads) const {
  VK_CHECK_COND(is_valid(), "Local workgroup dimensions must be nonzero");
  VK_CHECK_COND(
      x() <= max_lwg[0] && y() <= max_lwg[1] && z() <= max_lwg[2] &&
          nthreads() <= max_nthreads,
      "Local workgroup exceeds device limits");
}

void LocalWorkGroup::fit_to_global(const GlobalWorkGrid& gwg) {
  if (!target_lwg_shape_.is_valid()) {
    return;
  }

  utils::uvec3 exponent_limits{};
  bool requires_refit = false;

  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    exponent_limits[axis] =
        useful_exponent_limit(gwg.extents()[axis], target_total_nthreads_exp_);
    requires_refit =
        requires_refit || xyz_exponents_[axis] > exponent_limits[axis];
  }
  if (!requires_refit) {
    return;
  }

  utils::uvec3 exponents = xyz_exponents_;
  uint32_t excess_exponents = 0u;
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (exponents[axis] > exponent_limits[axis]) {
      excess_exponents += exponents[axis] - exponent_limits[axis];
      exponents[axis] = exponent_limits[axis];
    }
  }

  excess_exponents = target_lwg_shape_.allocate_exponents(
      excess_exponents, exponent_limits, exponents);
  allocate_by_exponent_capacity(excess_exponents, exponent_limits, exponents);

  xyz_exponents_ = exponents;
}

GlobalWorkGrid::GlobalWorkGrid(
    const utils::uvec3& extents,
    const DispatchGridIntent intent)
    : extents_(extents), intent_(intent), required_lwg_() {}

GlobalWorkGrid::GlobalWorkGrid(
    const utils::uvec3& extents,
    const DispatchGridIntent intent,
    const LocalWorkGroup& required_lwg)
    : extents_(extents), intent_(intent), required_lwg_(required_lwg) {}

bool GlobalWorkGrid::operator==(const GlobalWorkGrid& other) const {
  return extents_ == other.extents_ && intent_ == other.intent_ &&
      required_lwg_ == other.required_lwg_;
}

bool GlobalWorkGrid::operator!=(const GlobalWorkGrid& other) const {
  return !(*this == other);
}

const utils::uvec3& GlobalWorkGrid::extents() const {
  return extents_;
}

const LocalWorkGroup& GlobalWorkGrid::required_lwg_size() const {
  return required_lwg_;
}

DispatchGridIntent GlobalWorkGrid::intent() const {
  return intent_;
}

bool GlobalWorkGrid::is_linear() const {
  return intent_ == kLinearWorkGrid;
}

void GlobalWorkGrid::wrap_linear_dispatch(
    const utils::uvec3& max_wg_count,
    const uint32_t target_total_nthreads) {
  if (!is_linear() || required_lwg_.is_valid()) {
    return;
  }

  VK_CHECK_COND(
      extents_[1] == 1u && extents_[2] == 1u,
      "Linear dispatch wrapping requires one-dimensional input extents");
  VK_CHECK_COND(
      max_wg_count[0] > 0u && max_wg_count[1] > 0u,
      "Linear dispatch requires nonzero X and Y workgroup limits");

  const LocalWorkGroup required_lwg(kLinearLwg, target_total_nthreads);
  const uint64_t lwg_x = required_lwg.x();
  const uint64_t required_workgroups =
      utils::div_up(uint64_t(extents_[0]), lwg_x);
  if (required_workgroups <= max_wg_count[0]) {
    required_lwg_ = required_lwg;
    return;
  }

  const uint64_t square_width = static_cast<uint64_t>(
      std::ceil(std::sqrt(static_cast<double>(required_workgroups))));
  const uint64_t workgroups_x =
      std::min<uint64_t>(square_width, max_wg_count[0]);
  const uint64_t workgroups_y =
      utils::div_up(required_workgroups, workgroups_x);
  VK_CHECK_COND(
      workgroups_y <= max_wg_count[1],
      "Linear dispatch exceeds two-dimensional workgroup limits");

  extents_ = {
      utils::safe_downcast<uint32_t>(workgroups_x * lwg_x),
      utils::safe_downcast<uint32_t>(workgroups_y),
      1u};
  required_lwg_ = required_lwg;
}

} // namespace vkcompute
