/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/common_shims_slim.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Advanced indexing, self[indices], with the semantics of aten::index.Tensor.
 *
 * Inductor calls this fallback when an index is a mask, so the number of true
 * elements decides the output size. Reading that count waits for the stream,
 * so a CUDA graph cannot capture this call. As in ATen, an index out of range
 * triggers a device-side assert, and an index into a dimension of size 0 is
 * an error unless the indices select nothing.
 *
 * @param self Tensor to index (CUDA device). It, each index and the result
 * can have up to 16 dims.
 * @param indices indices_len_ pointers to index tensors, null for a full
 * slice. An index is int64 or int32, or a bool or uint8 mask that covers as
 * many dimensions of self as it has. At least one entry must be an index that
 * is not a 0-dim mask.
 * @param indices_len_ Number of entries in indices
 * @param ret0 Output: new tensor, strided the way Inductor expects
 * @return AOTITorchError
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_index_Tensor(
    Tensor* self,
    Tensor* const** indices,
    int64_t indices_len_,
    Tensor** ret0);

/**
 * Writes values into out at indices, with the semantics of aten::index_put_.
 * When accumulate is set, every value is added, also when an index repeats.
 *
 * Inductor calls this fallback, with out and self the same tensor, for an
 * index that is a mask, for accumulation into a dtype its generated code
 * cannot add atomically, and in deterministic mode. Writes that land on the
 * same element are applied in index order, and half and bfloat16 values are
 * summed in float, so the result is deterministic and small terms are not
 * lost. A mask makes this call wait for the stream. Indices are checked as
 * for aoti_torch_cuda_index_Tensor.
 *
 * @param out Tensor to write (CUDA device, up to 16 dims, as for
 * aoti_torch_cuda_index_Tensor)
 * @param self Must be out; AOTInductor passes the tensor it updates as both
 * @param indices num_indices index tensors, null for a full slice, with the
 * same dtypes as for aoti_torch_cuda_index_Tensor
 * @param num_indices Number of entries in indices
 * @param values Values to write, broadcastable to the indexed shape, same
 * dtype as out
 * @param accumulate If true, add values instead of writing them
 * @return AOTITorchError
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_index_put_out(
    Tensor* out,
    Tensor* self,
    Tensor* const* indices,
    const uint32_t num_indices,
    Tensor* values,
    bool accumulate);

/**
 * Reduces src into out at index along dim, with the semantics of
 * aten::scatter_reduce.two.
 *
 * Inductor calls this fallback, with out and self the same tensor, for every
 * reduction but "sum", for a sum into a dtype its generated code cannot add
 * atomically, and in deterministic mode. Values that land on the same element
 * are reduced in index order, so the result is deterministic. Half and
 * bfloat16 sums add in float, as in ATen's deterministic mode. A product
 * rounds to the dtype of out at every step and a mean rounds its sum before
 * dividing, as in ATen. As in ATen, an index out of range, negative ones
 * included, triggers a device-side assert.
 *
 * @param out Tensor to update (CUDA device, up to 16 dims)
 * @param self Must be out; AOTInductor passes the tensor it updates as both
 * @param dim Dimension of out that index addresses
 * @param index int64 tensor with as many dims as out, no larger than src, and
 * no larger than out except at dim
 * @param src Values to reduce, same dtype as out. Float, double, half,
 * bfloat16 and the integer dtypes are supported, and bool for "sum" only, as
 * in ATen.
 * @param reduce "sum", "prod", "mean", "amax" or "amin"
 * @param include_self If zero, the elements index names are reduced from src
 * alone; the others keep their values either way
 * @return AOTITorchError
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_scatter_reduce_two_out(
    Tensor* out,
    Tensor* self,
    int64_t dim,
    Tensor* index,
    Tensor* src,
    const char* reduce,
    int32_t include_self);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
