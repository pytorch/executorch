/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/util/size_util.h>

namespace executorch::backends::aoti::slim {

// =============================================================================
// compute_numel Tests
// =============================================================================

TEST(SizeUtilTest, ComputeNumel1D) {
  std::vector<int64_t> sizes = {10};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 10);
}

TEST(SizeUtilTest, ComputeNumel2D) {
  std::vector<int64_t> sizes = {3, 4};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 12);
}

TEST(SizeUtilTest, ComputeNumel3D) {
  std::vector<int64_t> sizes = {2, 3, 4};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 24);
}

TEST(SizeUtilTest, ComputeNumel4D) {
  std::vector<int64_t> sizes = {2, 3, 4, 5};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 120);
}

TEST(SizeUtilTest, ComputeNumelEmpty) {
  std::vector<int64_t> sizes = {0, 3, 4};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 0);
}

TEST(SizeUtilTest, ComputeNumelScalar) {
  std::vector<int64_t> sizes = {};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 1);
}

TEST(SizeUtilTest, ComputeNumelWithOnes) {
  std::vector<int64_t> sizes = {1, 1, 5, 1};
  EXPECT_EQ(compute_numel(makeArrayRef(sizes)), 5);
}

// =============================================================================
// compute_contiguous_strides Tests
// =============================================================================

TEST(SizeUtilTest, ComputeContiguousStrides1D) {
  std::vector<int64_t> sizes = {10};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 1u);
  EXPECT_EQ(strides[0], 1);
}

TEST(SizeUtilTest, ComputeContiguousStrides2D) {
  std::vector<int64_t> sizes = {3, 4};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 2u);
  EXPECT_EQ(strides[0], 4);
  EXPECT_EQ(strides[1], 1);
}

TEST(SizeUtilTest, ComputeContiguousStrides3D) {
  std::vector<int64_t> sizes = {2, 3, 4};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 3u);
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
}

TEST(SizeUtilTest, ComputeContiguousStrides4D) {
  std::vector<int64_t> sizes = {2, 3, 4, 5};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 4u);
  EXPECT_EQ(strides[0], 60);
  EXPECT_EQ(strides[1], 20);
  EXPECT_EQ(strides[2], 5);
  EXPECT_EQ(strides[3], 1);
}

TEST(SizeUtilTest, ComputeContiguousStridesScalar) {
  std::vector<int64_t> sizes = {};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 0u);
}

TEST(SizeUtilTest, ComputeContiguousStridesWithZero) {
  std::vector<int64_t> sizes = {0, 3, 4};
  auto strides = compute_contiguous_strides(makeArrayRef(sizes));

  EXPECT_EQ(strides.size(), 3u);
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
}

// =============================================================================
// compute_storage_nbytes_contiguous Tests
// =============================================================================

TEST(SizeUtilTest, ComputeStorageNbytesContiguousFloat) {
  std::vector<int64_t> sizes = {2, 3};
  size_t nbytes =
      compute_storage_nbytes_contiguous(makeArrayRef(sizes), sizeof(float), 0);
  EXPECT_EQ(nbytes, 6 * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesContiguousDouble) {
  std::vector<int64_t> sizes = {2, 3, 4};
  size_t nbytes =
      compute_storage_nbytes_contiguous(makeArrayRef(sizes), sizeof(double), 0);
  EXPECT_EQ(nbytes, 24 * sizeof(double));
}

TEST(SizeUtilTest, ComputeStorageNbytesContiguousWithOffset) {
  std::vector<int64_t> sizes = {2, 3};
  size_t nbytes =
      compute_storage_nbytes_contiguous(makeArrayRef(sizes), sizeof(float), 10);
  EXPECT_EQ(nbytes, (10 + 6) * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesContiguousEmpty) {
  std::vector<int64_t> sizes = {0, 3, 4};
  size_t nbytes =
      compute_storage_nbytes_contiguous(makeArrayRef(sizes), sizeof(float), 0);
  EXPECT_EQ(nbytes, 0u);
}

// =============================================================================
// compute_storage_nbytes (strided) Tests
// =============================================================================

TEST(SizeUtilTest, ComputeStorageNbytesContiguousTensor) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 0);
  EXPECT_EQ(nbytes, 6 * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesTransposedTensor) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {1, 2};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 0);
  EXPECT_EQ(nbytes, 6 * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesWithOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 5);
  EXPECT_EQ(nbytes, (5 + 6) * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesStrided3D) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 0);
  EXPECT_EQ(nbytes, 24 * sizeof(float));
}

TEST(SizeUtilTest, ComputeStorageNbytesEmpty) {
  std::vector<int64_t> sizes = {0, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 0);
  EXPECT_EQ(nbytes, 0u);
}

TEST(SizeUtilTest, ComputeStorageNbytesNonContiguous) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {4, 1};
  size_t nbytes = compute_storage_nbytes(
      makeArrayRef(sizes), makeArrayRef(strides), sizeof(float), 0);
  EXPECT_EQ(nbytes, 6 * sizeof(float));
}

// =============================================================================
// ArrayRefUtil Tests
// =============================================================================

TEST(ArrayRefUtilTest, MakeArrayRefFromVector) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  IntArrayRef ref = makeArrayRef(vec);

  EXPECT_EQ(ref.size(), 5u);
  EXPECT_EQ(ref[0], 1);
  EXPECT_EQ(ref[4], 5);
}

TEST(ArrayRefUtilTest, MakeArrayRefFromVectorConstruction) {
  std::vector<int64_t> values = {10, 20, 30};
  IntArrayRef ref = makeArrayRef(values);

  EXPECT_EQ(ref.size(), 3u);
  EXPECT_EQ(ref[0], 10);
  EXPECT_EQ(ref[2], 30);
}

TEST(ArrayRefUtilTest, ToVec) {
  std::vector<int64_t> original = {1, 2, 3};
  IntArrayRef ref = makeArrayRef(original);
  std::vector<int64_t> copy = toVec(ref);

  EXPECT_EQ(copy.size(), 3u);
  EXPECT_EQ(copy[0], 1);
  EXPECT_EQ(copy[2], 3);

  copy[0] = 100;
  EXPECT_EQ(original[0], 1);
}

} // namespace executorch::backends::aoti::slim
