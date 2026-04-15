/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/sort.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

namespace c10_slim = executorch::backends::aoti::slim::c10;

// PyTorch ScalarType::Half = 5, not defined in slim ScalarType enum.
constexpr auto kHalf = static_cast<c10_slim::ScalarType>(5);

namespace {

__global__ void init_indices_kernel(
    int64_t* data,
    int64_t slice_size,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    data[idx] = idx % slice_size;
  }
}

// Permute between [outer, sort, inner] and [outer, inner, sort] layouts.
// forward=true:  src is [outer, sort, inner], dst is [outer, inner, sort]
// forward=false: src is [outer, inner, sort], dst is [outer, sort, inner]
template <typename T>
__global__ void permute_sort_dim_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t sort_size,
    int64_t inner_size,
    int64_t total_elements,
    bool forward) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;
  // Decompose idx in [outer, inner, sort] layout
  int64_t s = idx % sort_size;
  int64_t i = (idx / sort_size) % inner_size;
  int64_t o = idx / (sort_size * inner_size);
  // Corresponding offset in [outer, sort, inner] layout
  int64_t other = o * sort_size * inner_size + s * inner_size + i;
  if (forward) {
    dst[idx] = src[other];
  } else {
    dst[other] = src[idx];
  }
}

void launch_permute(
    const void* src,
    void* dst,
    int64_t sort_size,
    int64_t inner_size,
    int64_t elem_size,
    int64_t total_elements,
    bool forward,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = static_cast<int>((total_elements + threads - 1) / threads);
  switch (elem_size) {
    case 8:
      permute_sort_dim_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<const int64_t*>(src),
          static_cast<int64_t*>(dst),
          sort_size,
          inner_size,
          total_elements,
          forward);
      break;
    case 4:
      permute_sort_dim_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<const int32_t*>(src),
          static_cast<int32_t*>(dst),
          sort_size,
          inner_size,
          total_elements,
          forward);
      break;
    case 2:
      permute_sort_dim_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<const int16_t*>(src),
          static_cast<int16_t*>(dst),
          sort_size,
          inner_size,
          total_elements,
          forward);
      break;
  }
}

template <typename T>
void sort_slice_impl(
    T* keys,
    int64_t* values,
    int64_t n,
    bool descending,
    bool stable,
    cudaStream_t stream) {
  auto k = thrust::device_pointer_cast(keys);
  auto v = thrust::device_pointer_cast(values);
  if (stable && descending) {
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v, thrust::greater<T>());
  } else if (stable) {
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v);
  } else if (descending) {
    thrust::sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v, thrust::greater<T>());
  } else {
    thrust::sort_by_key(thrust::cuda::par.on(stream), k, k + n, v);
  }
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_sort_stable(
    Tensor* self,
    int32_t* stable,
    int64_t dim,
    int32_t descending,
    Tensor** ret0,
    Tensor** ret1) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: ret0 is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret1 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: ret1 is null");

  int64_t ndim = static_cast<int64_t>(self->dim());

  if (dim < 0)
    dim += ndim;
  ET_CHECK_OR_RETURN_ERROR(
      dim >= 0 && dim < ndim,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: dim out of range");

  ET_CHECK_OR_RETURN_ERROR(
      self->is_contiguous(),
      NotSupported,
      "aoti_torch_cuda_sort_stable: non-contiguous input not supported");

  // Validate dtype early and compute element size (needed for transpose path).
  auto self_dtype = self->dtype();
  int64_t elem_size = 0;
  switch (self_dtype) {
    case c10_slim::ScalarType::Long:
      elem_size = sizeof(int64_t);
      break;
    case c10_slim::ScalarType::Int:
      elem_size = sizeof(int32_t);
      break;
    case c10_slim::ScalarType::Float:
      elem_size = sizeof(float);
      break;
    case c10_slim::ScalarType::BFloat16:
      elem_size = sizeof(__nv_bfloat16);
      break;
    case kHalf:
      elem_size = sizeof(__half);
      break;
    default:
      ET_LOG(
          Error,
          "aoti_torch_cuda_sort_stable: unsupported dtype %d",
          static_cast<int>(self_dtype));
      return Error::InvalidArgument;
  }

  int64_t sort_size = self->size(dim);
  int64_t total_elements = static_cast<int64_t>(self->numel());
  int64_t num_slices = (sort_size > 0) ? total_elements / sort_size : 0;

  int32_t device_idx = static_cast<int32_t>(self->device_index());

  auto stream_result = getCurrentCUDAStream(self->device_index());
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_cuda_sort_stable: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  // Contiguous strides for output tensors
  auto input_sizes = self->sizes();
  std::vector<int64_t> contig_strides(ndim);
  if (ndim > 0) {
    contig_strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
      contig_strides[i] = contig_strides[i + 1] * input_sizes[i + 1];
    }
  }

  int32_t dtype_val = static_cast<int32_t>(self_dtype);

  // Allocate output values (same shape/dtype as input)
  *ret0 = nullptr;
  aoti_torch_empty_strided(
      ndim,
      input_sizes.data(),
      contig_strides.data(),
      dtype_val,
      static_cast<int32_t>(c10_slim::DeviceType::CUDA),
      device_idx,
      ret0);
  ET_CHECK_OR_RETURN_ERROR(
      *ret0 != nullptr,
      Internal,
      "aoti_torch_cuda_sort_stable: failed to allocate values tensor");

  // Copy input data to output values
  if (total_elements > 0) {
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
        (*ret0)->data_ptr(),
        self->data_ptr(),
        self->nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));
  }

  // Allocate output indices (same shape, int64 dtype)
  *ret1 = nullptr;
  aoti_torch_empty_strided(
      ndim,
      input_sizes.data(),
      contig_strides.data(),
      static_cast<int32_t>(c10_slim::ScalarType::Long),
      static_cast<int32_t>(c10_slim::DeviceType::CUDA),
      device_idx,
      ret1);
  ET_CHECK_OR_RETURN_ERROR(
      *ret1 != nullptr,
      Internal,
      "aoti_torch_cuda_sort_stable: failed to allocate indices tensor");

  if (sort_size <= 1) {
    if (total_elements > 0) {
      int threads = 256;
      int blocks = static_cast<int>((total_elements + threads - 1) / threads);
      init_indices_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<int64_t*>((*ret1)->data_ptr()),
          sort_size,
          total_elements);
      ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
    }
    return Error::Ok;
  }

  bool is_stable = (stable != nullptr && *stable != 0);
  bool desc = (descending != 0);

  int64_t dim_stride = self->stride(dim);
  bool needs_transpose = (dim_stride != 1 && ndim > 1);

  // For innermost dim, sort ret0/ret1 directly. For non-innermost dim,
  // transpose to make sort dim last, sort contiguous slices, then scatter back.
  void* values_base = nullptr;
  int64_t* indices_base = nullptr;
  void* temp_values_buf = nullptr;
  void* temp_indices_buf = nullptr;
  int64_t inner_size = 1;

  if (needs_transpose) {
    for (int64_t d = dim + 1; d < ndim; ++d) {
      inner_size *= input_sizes[d];
    }

    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocAsync(
        &temp_values_buf,
        static_cast<size_t>(total_elements * elem_size),
        stream));
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocAsync(
        &temp_indices_buf,
        static_cast<size_t>(total_elements) * sizeof(int64_t),
        stream));

    // Gather: [outer, sort, inner] → [outer, inner, sort]
    launch_permute(
        (*ret0)->data_ptr(),
        temp_values_buf,
        sort_size,
        inner_size,
        elem_size,
        total_elements,
        true,
        stream);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    init_indices_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int64_t*>(temp_indices_buf),
        sort_size,
        total_elements);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

    values_base = temp_values_buf;
    indices_base = static_cast<int64_t*>(temp_indices_buf);
  } else {
    if (total_elements > 0) {
      int threads = 256;
      int blocks = static_cast<int>((total_elements + threads - 1) / threads);
      init_indices_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<int64_t*>((*ret1)->data_ptr()),
          sort_size,
          total_elements);
      ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
    }

    values_base = (*ret0)->data_ptr();
    indices_base = static_cast<int64_t*>((*ret1)->data_ptr());
  }

  for (int64_t s = 0; s < num_slices; ++s) {
    int64_t offset = s * sort_size;
    int64_t* idx_ptr = indices_base + offset;

    switch (self_dtype) {
      case c10_slim::ScalarType::Long: {
        sort_slice_impl(
            static_cast<int64_t*>(values_base) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case c10_slim::ScalarType::Int: {
        sort_slice_impl(
            static_cast<int32_t*>(values_base) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case c10_slim::ScalarType::Float: {
        sort_slice_impl(
            static_cast<float*>(values_base) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case c10_slim::ScalarType::BFloat16: {
        sort_slice_impl(
            static_cast<__nv_bfloat16*>(values_base) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case kHalf: {
        sort_slice_impl(
            static_cast<__half*>(values_base) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      default:
        break;
    }
  }

  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

  if (needs_transpose) {
    // Scatter: [outer, inner, sort] → [outer, sort, inner]
    launch_permute(
        temp_values_buf,
        (*ret0)->data_ptr(),
        sort_size,
        inner_size,
        elem_size,
        total_elements,
        false,
        stream);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

    launch_permute(
        temp_indices_buf,
        (*ret1)->data_ptr(),
        sort_size,
        inner_size,
        static_cast<int64_t>(sizeof(int64_t)),
        total_elements,
        false,
        stream);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaFreeAsync(temp_values_buf, stream));
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaFreeAsync(temp_indices_buf, stream));
  }

  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
