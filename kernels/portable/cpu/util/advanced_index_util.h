/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;
using TensorOptList = exec_aten::ArrayRef<exec_aten::optional<Tensor>>;

/**
 * Performs preliminary checks on the arguments. However, it doesn't check that
 * the values of integer indices are within the right bounds for the given
 * input tensor
 */
bool check_index_args(const Tensor& in, TensorOptList indices, Tensor& out);

/**
 * The output shape depends on whether there are null indices in between
 * non-null indices or not. So, conceptually, the list of indices can be
 * divided in alternating segments of non-null indices and null indices.
 * We refer to the segments of non-null indices as blocks. If the indices list
 * has 0 blocks, it means that the list is empty, or all its elements are null.
 * If the list has exactly 1 block, it means that all the non-null indices are
 * contiguous, and there are possibly some null indices at the beginning of the
 * list and some of at the end. If the list has more than 1 block, it means
 * there are null indices in between the non-null inidces.
 * This functions simplu counts the number of blocks (i.e. non-null segments) in
 * the indices list.
 */
size_t count_index_blocks(TensorOptList indices);

/**
 * Counts the number of true values in a mask index
 */
size_t count_trues_in_mask_index(const Tensor& index);

/**
 * Compute the broadcast shape between the indices
 */
bool get_indices_broadcast_shape(
    TensorOptList indices,
    Tensor::SizesType* ix_sizes,
    size_t* ix_ndim);

/**
 * Compute the dimension of the broadcast shape between the indices
 */
size_t get_indices_broadcast_ndim(TensorOptList indices);

/**
 * Computes the number of dimensions that are being indexed by some non-null
 * index.
 */
size_t get_num_indexed_dims(TensorOptList indices);

/**
 * Computes the number of null indices
 */
size_t get_num_null_indices(TensorOptList indices);

/**
 * Computes the number of null indices at the beginning of the list
 */
size_t get_num_leading_null_indices(TensorOptList indices);

/**
 * Compute the expected size for the out tensor
 */
bool get_index_out_target_size(
    const Tensor& in,
    TensorOptList indices,
    bool adjacent,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

/**
 * dim_map maps non-indexed input dimensions to the corresponding output
 * dimensions. Indexed dimensions are mapped to -1.
 */
void compute_dim_map(
    const Tensor& in,
    TensorOptList indices,
    int32_t* dim_map,
    bool adjacent);

/**
 * ix_map maps indexed input dimensions to the corresponding index.
 * Non-indexed dimensions are mapped to -1.
 */
void compute_index_map(
    const Tensor& in,
    TensorOptList indices,
    int32_t* ix_map);

/**
 * Computes the input coordinate corresponding to a given output coordinate
 */
bool get_in_coord(
    const Tensor& in,
    TensorOptList indices,
    size_t start,
    size_t broadcast_ndim,
    int32_t* dim_map,
    int32_t* ix_map,
    size_t* out_coord,
    size_t* in_coord);

/**
 * Computes input flat index corresponding to a given output flat index
 */
std::pair<size_t, bool> get_in_ix(
    const Tensor& in,
    TensorOptList indices,
    Tensor& out,
    size_t out_ix,
    size_t start,
    size_t broadcast_ndim,
    int32_t* dim_map,
    int32_t* ix_map);

} // namespace executor
} // namespace torch
