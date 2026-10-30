/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include <c10/macros/Macros.h>

#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/cuda/runtime/shims/index.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

namespace c10_slim = executorch::backends::aoti::slim::c10;
using c10_slim::ScalarType;
using executorch::runtime::Error;

namespace {

constexpr int kMaxDims = 16;
constexpr int kThreads = 256;

// One indexed dimension of the tensor being indexed. Its index is an int64 or
// int32 tensor, or the row-major positions of a mask's true elements, from
// which this dimension's coordinate is (position / mask_divisor) % mask_size.
struct DimIndex {
  const void* data;
  int64_t strides[kMaxDims]; // along the iteration dims, 0 where broadcast
  int64_t size; // of the indexed dimension
  int64_t stride; // of the indexed dimension
  int64_t mask_divisor; // 0 for an index tensor
  int64_t mask_size;
  bool is_int32;
};

// The iteration space of an indexing call is the shape of the indexed result.
// Each point maps to an element of the indexed tensor and to one of the other
// tensor: the output of a gather, the values of a put. The dims the indices
// produce are [broadcast_start, broadcast_start + broadcast_ndim).
struct IndexPlan {
  int ndim;
  int64_t sizes[kMaxDims];
  int64_t indexed_strides[kMaxDims]; // 0 along the dims the indices produce
  int64_t other_strides[kMaxDims];
  int broadcast_start;
  int broadcast_ndim;
  int64_t broadcast_numel;
  int64_t after_numel; // of the dims after the ones the indices produce
  bool from_mask[kMaxDims]; // size is a mask's count, unknown at export
  int num_indexed;
  DimIndex indexed[kMaxDims];
};

// Adds the offsets of `linear` within dims [first, end).
__device__ __forceinline__ void add_offsets(
    const IndexPlan& plan,
    int first,
    int end,
    int64_t linear,
    int64_t& indexed_offset,
    int64_t& other_offset) {
  for (int d = end - 1; d >= first; --d) {
    const int64_t coordinate = linear % plan.sizes[d];
    linear /= plan.sizes[d];
    indexed_offset += coordinate * plan.indexed_strides[d];
    other_offset += coordinate * plan.other_strides[d];
  }
}

// Adds the offsets of a point of the dims the indices produce, including the
// element the indices pick.
__device__ __forceinline__ void add_index_offsets(
    const IndexPlan& plan,
    int64_t point,
    int64_t& indexed_offset,
    int64_t& other_offset) {
  int64_t index_offsets[kMaxDims];
  for (int i = 0; i < plan.num_indexed; ++i) {
    index_offsets[i] = 0;
  }
  for (int d = plan.broadcast_start + plan.broadcast_ndim - 1;
       d >= plan.broadcast_start;
       --d) {
    const int64_t coordinate = point % plan.sizes[d];
    point /= plan.sizes[d];
    other_offset += coordinate * plan.other_strides[d];
    for (int i = 0; i < plan.num_indexed; ++i) {
      index_offsets[i] += coordinate * plan.indexed[i].strides[d];
    }
  }
  for (int i = 0; i < plan.num_indexed; ++i) {
    const DimIndex& index = plan.indexed[i];
    int64_t position = index.is_int32
        ? static_cast<const int32_t*>(index.data)[index_offsets[i]]
        : static_cast<const int64_t*>(index.data)[index_offsets[i]];
    if (index.mask_divisor != 0) {
      position = position / index.mask_divisor % index.mask_size;
    }
    if (position < 0) {
      position += index.size;
    }
    // Checked on the device, as ATen and Inductor's generated code do, since
    // a check on the host would have to wait for the stream.
    CUDA_KERNEL_ASSERT_MSG(
        position >= 0 && position < index.size, "index out of range");
    indexed_offset += position * index.stride;
  }
}

// Adds the offsets of a point of the other dims, the slice each index picks.
__device__ __forceinline__ void add_slice_offsets(
    const IndexPlan& plan,
    int64_t point,
    int64_t& indexed_offset,
    int64_t& other_offset) {
  const int end = plan.broadcast_start + plan.broadcast_ndim;
  add_offsets(
      plan,
      end,
      plan.ndim,
      point % plan.after_numel,
      indexed_offset,
      other_offset);
  add_offsets(
      plan,
      0,
      plan.broadcast_start,
      point / plan.after_numel,
      indexed_offset,
      other_offset);
}

__device__ __forceinline__ void locate(
    const IndexPlan& plan,
    int64_t linear,
    int64_t& indexed_offset,
    int64_t& other_offset) {
  indexed_offset = 0;
  other_offset = 0;
  const int64_t rest = linear / plan.after_numel;
  add_index_offsets(
      plan, rest % plan.broadcast_numel, indexed_offset, other_offset);
  add_slice_offsets(
      plan,
      rest / plan.broadcast_numel * plan.after_numel +
          linear % plan.after_numel,
      indexed_offset,
      other_offset);
}

template <typename T>
__global__ void gather_kernel(
    const IndexPlan plan,
    int64_t numel,
    const T* __restrict__ self,
    T* __restrict__ out) {
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < numel;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t self_offset;
    int64_t out_offset;
    locate(plan, linear, self_offset, out_offset);
    out[out_offset] = self[self_offset];
  }
}

// Adds in AccT, ATen's accumulation type, and rounds once.
template <typename T, typename AccT, bool kAccumulate>
__device__ __forceinline__ void store(T* out, AccT sum) {
  *out = kAccumulate ? static_cast<T>(static_cast<AccT>(*out) + sum)
                     : static_cast<T>(sum);
}

// For indices that cannot repeat an element, which is any mix of masks and
// full slices.
template <typename T, typename AccT, bool kAccumulate>
__global__ void put_kernel(
    const IndexPlan plan,
    int64_t numel,
    T* __restrict__ out,
    const T* __restrict__ values) {
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < numel;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t out_offset;
    int64_t values_offset;
    locate(plan, linear, out_offset, values_offset);
    store<T, AccT, kAccumulate>(
        out + out_offset, static_cast<AccT>(values[values_offset]));
  }
}

__global__ void index_offsets_kernel(
    const IndexPlan plan,
    int64_t points,
    int64_t* __restrict__ out_offsets,
    int64_t* __restrict__ values_offsets) {
  for (int64_t point =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < points;
       point += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t out_offset = 0;
    int64_t values_offset = 0;
    add_index_offsets(plan, point, out_offset, values_offset);
    out_offsets[point] = out_offset;
    values_offsets[point] = values_offset;
  }
}

// After a stable sort of the index points by the element they pick, the
// points that pick one element are a run in index order. Each thread takes
// one element of the slice of one run, adds the run's values in that order
// or keeps the last, and writes once, so the result does not depend on
// scheduling.
template <typename T, typename AccT, bool kAccumulate>
__global__ void put_sorted_kernel(
    const IndexPlan plan,
    int64_t points,
    int64_t slice_numel,
    const int64_t* __restrict__ out_offsets,
    const int64_t* __restrict__ values_offsets,
    T* __restrict__ out,
    const T* __restrict__ values) {
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < points * slice_numel;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t i = linear / slice_numel;
    const int64_t offset = out_offsets[i];
    if (i > 0 && out_offsets[i - 1] == offset) {
      continue;
    }
    int64_t out_offset = offset;
    int64_t values_slice_offset = 0;
    add_slice_offsets(
        plan, linear % slice_numel, out_offset, values_slice_offset);
    // Starts from the element itself, so the additions associate as they do
    // one after another.
    AccT result = kAccumulate ? static_cast<AccT>(out[out_offset]) : AccT{};
    for (int64_t j = i; j < points && out_offsets[j] == offset; ++j) {
      const auto value =
          static_cast<AccT>(values[values_offsets[j] + values_slice_offset]);
      result = kAccumulate ? static_cast<AccT>(result + value) : value;
    }
    out[out_offset] = static_cast<T>(result);
  }
}

// A mask's element at a row-major position is set.
struct MaskIsSet {
  const uint8_t* data;
  int ndim;
  int64_t sizes[kMaxDims];
  int64_t strides[kMaxDims];

  __device__ bool operator()(int64_t position) const {
    int64_t offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
      offset += position % sizes[d] * strides[d];
      position /= sizes[d];
    }
    return data[offset] != 0;
  }
};

// Stream-ordered scratch, freed on the same stream when it goes out of scope.
class Scratch {
 public:
  Scratch(int32_t device, cudaStream_t stream)
      : device_(device), stream_(stream) {}

  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;

  ~Scratch() {
    for (void* block : blocks_) {
      (void)CudaAllocator::deallocate_stream_ordered(block, device_, stream_);
    }
  }

  void* allocate(size_t nbytes) {
    auto block = CudaAllocator::allocate_stream_ordered(
        std::max<size_t>(nbytes, 1), device_, stream_);
    if (!block.ok()) {
      return nullptr;
    }
    blocks_.push_back(block.get());
    return block.get();
  }

 private:
  int32_t device_;
  cudaStream_t stream_;
  std::vector<void*> blocks_;
};

// Writes the row-major positions of a mask's true elements to scratch and
// reads their count back, which waits for the stream.
Error mask_positions(
    const Tensor& mask,
    Scratch& scratch,
    cudaStream_t stream,
    const int64_t** positions,
    int64_t* count) {
  const auto numel = static_cast<int64_t>(mask.numel());
  *positions = nullptr;
  *count = 0;
  if (numel == 0) {
    return Error::Ok;
  }
  // A graph capture cannot record the host read of the count below, so refuse
  // before allocating anything.
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamIsCapturing(stream, &capture));
  ET_CHECK_OR_RETURN_ERROR(
      capture == cudaStreamCaptureStatusNone,
      NotSupported,
      "index: a mask index cannot run inside a CUDA graph capture");
  // CUB counts items in an int on CUDA 12.
  ET_CHECK_OR_RETURN_ERROR(
      numel <= INT_MAX,
      NotSupported,
      "index: masks with more than %d elements are not supported",
      INT_MAX);
  MaskIsSet is_set{static_cast<const uint8_t*>(mask.data_ptr())};
  is_set.ndim = static_cast<int>(mask.dim());
  for (int d = 0; d < is_set.ndim; ++d) {
    is_set.sizes[d] = mask.size(d);
    is_set.strides[d] = mask.stride(d);
  }
  auto* selected =
      static_cast<int64_t*>(scratch.allocate(numel * sizeof(int64_t)));
  auto* selected_count =
      static_cast<int64_t*>(scratch.allocate(sizeof(int64_t)));
  ET_CHECK_OR_RETURN_ERROR(
      selected != nullptr && selected_count != nullptr,
      MemoryAllocationFailed,
      "index: allocating the mask positions failed");
  thrust::counting_iterator<int64_t> first(0);
  size_t temp_bytes = 0;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cub::DeviceSelect::If(
      nullptr,
      temp_bytes,
      first,
      selected,
      selected_count,
      numel,
      is_set,
      stream));
  void* temp = scratch.allocate(temp_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      temp != nullptr,
      MemoryAllocationFailed,
      "index: allocating the mask scan scratch failed");
  ET_CUDA_CHECK_OR_RETURN_ERROR(cub::DeviceSelect::If(
      temp,
      temp_bytes,
      first,
      selected,
      selected_count,
      numel,
      is_set,
      stream));
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
      count, selected_count, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamSynchronize(stream));
  *positions = selected;
  return Error::Ok;
}

struct Advanced {
  int64_t dim; // of the indexed tensor
  bool from_mask;
  DimIndex index;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

// Lays out an indexing call the way ATen's index and index_put do: masks
// become one index per dimension they cover, all indices broadcast together,
// and the dimensions they produce replace the indexed ones if those are
// adjacent and go first otherwise. Sets *may_repeat when two points can land
// on one element, which only an index tensor allows.
Error plan_indexing(
    const Tensor& indexed,
    Tensor* const* indices,
    int64_t num_indices,
    Scratch& scratch,
    cudaStream_t stream,
    IndexPlan* plan,
    bool* may_repeat) {
  const auto ndim = static_cast<int64_t>(indexed.dim());
  ET_CHECK_OR_RETURN_ERROR(
      ndim <= kMaxDims,
      NotSupported,
      "index: tensors with more than %d dims are not supported",
      kMaxDims);
  ET_CHECK_OR_RETURN_ERROR(
      num_indices <= ndim,
      InvalidArgument,
      "index: too many indices for a tensor of %" PRId64 " dims",
      ndim);
  *may_repeat = false;
  std::vector<Advanced> advanced;
  int64_t dim = 0;
  for (int64_t i = 0; i < num_indices; ++i) {
    const Tensor* index = indices[i];
    if (index == nullptr) {
      ++dim;
      continue;
    }
    ET_CHECK_OR_RETURN_ERROR(
        index->is_cuda(),
        NotSupported,
        "index: index %" PRId64 " is not on the CUDA device",
        i);
    const auto dtype = index->dtype();
    if (dtype == ScalarType::Bool || dtype == ScalarType::Byte) {
      const auto mask_ndim = static_cast<int64_t>(index->dim());
      // A 0-dim mask covers no dims, so it adds no index, as in ATen.
      if (mask_ndim == 0) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          dim + mask_ndim <= ndim,
          InvalidArgument,
          "index: too many indices for a tensor of %" PRId64 " dims",
          ndim);
      for (int64_t j = 0; j < mask_ndim; ++j) {
        ET_CHECK_OR_RETURN_ERROR(
            index->size(j) == indexed.size(dim + j),
            InvalidArgument,
            "index: mask %" PRId64
            " does not match the indexed shape at dim %" PRId64,
            i,
            dim + j);
      }
      const int64_t* positions = nullptr;
      int64_t count = 0;
      ET_CHECK_OK_OR_RETURN_ERROR(
          mask_positions(*index, scratch, stream, &positions, &count));
      // Row-major position p has coordinate p / divisor % size along dim j.
      std::vector<int64_t> divisors(mask_ndim, 1);
      for (int64_t j = mask_ndim - 1; j > 0; --j) {
        divisors[j - 1] = divisors[j] * std::max<int64_t>(index->size(j), 1);
      }
      for (int64_t j = 0; j < mask_ndim; ++j) {
        Advanced entry{dim + j, true};
        entry.index.data = positions;
        entry.index.mask_divisor = divisors[j];
        entry.index.mask_size = index->size(j);
        entry.sizes = {count};
        entry.strides = {1};
        advanced.push_back(entry);
      }
      dim += mask_ndim;
    } else if (dtype == ScalarType::Long || dtype == ScalarType::Int) {
      ET_CHECK_OR_RETURN_ERROR(
          dim < ndim,
          InvalidArgument,
          "index: too many indices for a tensor of %" PRId64 " dims",
          ndim);
      ET_CHECK_OR_RETURN_ERROR(
          static_cast<int64_t>(index->dim()) <= kMaxDims,
          NotSupported,
          "index: index tensors with more than %d dims are not supported",
          kMaxDims);
      Advanced entry{dim, false};
      entry.index.data = index->data_ptr();
      entry.index.is_int32 = dtype == ScalarType::Int;
      entry.sizes.assign(index->sizes().begin(), index->sizes().end());
      entry.strides.assign(index->strides().begin(), index->strides().end());
      advanced.push_back(entry);
      *may_repeat = true;
      ++dim;
    } else {
      ET_LOG(
          Error,
          "index: index %" PRId64
          " has dtype %d; indices must be int64, int32, bool or uint8",
          i,
          static_cast<int>(dtype));
      return Error::InvalidArgument;
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      dim <= ndim,
      InvalidArgument,
      "index: too many indices for a tensor of %" PRId64 " dims",
      ndim);
  // ATen's advanced indexing rejects a call with only None and 0-dim masks;
  // its plan would cover the whole tensor.
  ET_CHECK_OR_RETURN_ERROR(
      !advanced.empty(),
      InvalidArgument,
      "index: at least one index must be a tensor that is not a 0-dim mask");

  size_t broadcast_ndim = 0;
  for (const auto& entry : advanced) {
    broadcast_ndim = std::max(broadcast_ndim, entry.sizes.size());
  }
  std::vector<int64_t> broadcast(broadcast_ndim, 1);
  std::vector<bool> masked(broadcast_ndim, false);
  std::vector<bool> fixed(broadcast_ndim, false);
  for (const auto& entry : advanced) {
    const size_t lead = broadcast_ndim - entry.sizes.size();
    for (size_t d = 0; d < entry.sizes.size(); ++d) {
      if (entry.from_mask) {
        masked[lead + d] = true;
      } else if (entry.sizes[d] != 1) {
        fixed[lead + d] = true;
      }
      int64_t& size = broadcast[lead + d];
      ET_CHECK_OR_RETURN_ERROR(
          entry.sizes[d] == 1 || size == 1 || entry.sizes[d] == size,
          InvalidArgument,
          "index: the indices cannot be broadcast together");
      if (size == 1) {
        size = entry.sizes[d];
      }
    }
  }

  const bool adjacent = advanced.back().dim - advanced.front().dim + 1 ==
      static_cast<int64_t>(advanced.size());
  const int64_t first = advanced.front().dim;
  std::vector<bool> is_indexed(ndim, false);
  for (const auto& entry : advanced) {
    is_indexed[entry.dim] = true;
  }
  std::vector<int64_t> plain_dims;
  for (int64_t d = 0; d < ndim; ++d) {
    if (!is_indexed[d]) {
      plain_dims.push_back(d);
    }
  }
  // Dims of the indexed tensor in iteration order, -1 for each broadcast dim.
  std::vector<int64_t> order;
  for (int64_t d : plain_dims) {
    if (adjacent && d < first) {
      order.push_back(d);
    }
  }
  const auto broadcast_start = static_cast<int64_t>(order.size());
  order.insert(order.end(), broadcast_ndim, -1);
  for (int64_t d : plain_dims) {
    if (!adjacent || d > first) {
      order.push_back(d);
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      order.size() <= static_cast<size_t>(kMaxDims),
      NotSupported,
      "index: results with more than %d dims are not supported",
      kMaxDims);

  *plan = IndexPlan{};
  plan->ndim = static_cast<int>(order.size());
  plan->broadcast_start = static_cast<int>(broadcast_start);
  plan->broadcast_ndim = static_cast<int>(broadcast_ndim);
  plan->broadcast_numel = 1;
  for (int64_t size : broadcast) {
    plan->broadcast_numel *= size;
  }
  // As in ATen, an index into a dim of size 0 is out of range unless there are
  // no index points. The result is empty, so no kernel would assert it.
  for (const auto& entry : advanced) {
    ET_CHECK_OR_RETURN_ERROR(
        plan->broadcast_numel == 0 || indexed.size(entry.dim) > 0,
        InvalidArgument,
        "index: index out of range for dim %" PRId64 " of size 0",
        entry.dim);
  }
  plan->after_numel = 1;
  for (size_t k = static_cast<size_t>(broadcast_start) + broadcast_ndim;
       k < order.size();
       ++k) {
    plan->after_numel *= indexed.size(order[k]);
  }
  for (size_t k = 0; k < order.size(); ++k) {
    if (order[k] >= 0) {
      plan->sizes[k] = indexed.size(order[k]);
      plan->indexed_strides[k] = indexed.stride(order[k]);
    } else {
      plan->sizes[k] = broadcast[k - broadcast_start];
      plan->from_mask[k] =
          masked[k - broadcast_start] && !fixed[k - broadcast_start];
    }
  }
  plan->num_indexed = static_cast<int>(advanced.size());
  for (size_t i = 0; i < advanced.size(); ++i) {
    const auto& entry = advanced[i];
    DimIndex& index = plan->indexed[i];
    index = entry.index;
    index.size = indexed.size(entry.dim);
    index.stride = indexed.stride(entry.dim);
    const size_t lead = broadcast_ndim - entry.sizes.size();
    for (size_t d = 0; d < entry.sizes.size(); ++d) {
      if (entry.sizes[d] != 1) {
        index.strides[broadcast_start + lead + d] = entry.strides[d];
      }
    }
  }
  return Error::Ok;
}

int64_t plan_numel(const IndexPlan& plan) {
  int64_t numel = 1;
  for (int d = 0; d < plan.ndim; ++d) {
    numel *= plan.sizes[d];
  }
  return numel;
}

unsigned int grid_size(int64_t numel) {
  return static_cast<unsigned int>(
      std::min<int64_t>((numel + kThreads - 1) / kThreads, 65535));
}

// The meta function decided the layout with each mask's count unknown, so a
// count is never taken to be 1, and the result is only known to have fewer
// than 2 elements when another size is 0.
bool is_one(const IndexPlan& plan, int d) {
  return !plan.from_mask[d] && plan.sizes[d] == 1;
}

bool is_contiguous(const IndexPlan& plan, const int64_t* strides) {
  bool any_unknown = false;
  int64_t numel = 1;
  for (int d = 0; d < plan.ndim; ++d) {
    if (!plan.from_mask[d] && plan.sizes[d] == 0) {
      return true;
    }
    any_unknown |= plan.from_mask[d];
    numel *= plan.sizes[d];
  }
  if (!any_unknown && numel < 2) {
    return true;
  }
  int64_t expected = 1;
  for (int d = plan.ndim - 1; d >= 0; --d) {
    if (is_one(plan, d)) {
      continue;
    }
    if (strides[d] != expected) {
      return false;
    }
    expected *= plan.sizes[d];
  }
  return true;
}

bool is_channels_last_contiguous(
    const IndexPlan& plan,
    const int64_t* strides) {
  if (plan.ndim != 4) {
    return false;
  }
  int64_t expected = 1;
  for (int d : {1, 3, 2, 0}) {
    if (!is_one(plan, d)) {
      if (strides[d] != expected) {
        return false;
      }
      expected *= plan.sizes[d];
    }
  }
  return true;
}

// The output strides index.Tensor's meta function gives, which Inductor bakes
// into the code that reads the result: those of an elementwise output for an
// input with `strides` (compute_elementwise_output_logical_to_physical_perm in
// torch/_prims_common).
std::vector<int64_t> output_strides(const IndexPlan& plan, bool self_is_empty) {
  const int64_t* sizes = plan.sizes;
  const int64_t* strides = plan.indexed_strides;
  const int ndim = plan.ndim;
  std::vector<int> perm(ndim);
  std::iota(perm.begin(), perm.end(), 0);
  const bool contiguous = is_contiguous(plan, strides);
  const bool channels_last = is_channels_last_contiguous(plan, strides);
  if (self_is_empty || ndim < 2 || (contiguous && !channels_last)) {
    // Row-major.
  } else if (channels_last && !contiguous) {
    perm = {0, 2, 3, 1};
  } else {
    auto ge = [](int64_t a, int64_t b) {
      return b == 0 || (a != 0 && (a >= b || a % b == 0));
    };
    auto should_swap = [&](int a, int b) {
      const int64_t stride_a = strides[a];
      const int64_t stride_b = strides[b];
      if (stride_a == 0 || stride_b == 0) {
        return 0;
      }
      if (stride_a == stride_b) {
        return ge(sizes[b], sizes[a]) ? 0 : 1;
      }
      if (ge(stride_b, stride_a)) {
        return -1;
      }
      return ge(stride_a, stride_b) ? 1 : 0;
    };
    std::reverse(perm.begin(), perm.end());
    for (int i = 1; i < ndim; ++i) {
      int dim1 = i;
      for (int dim0 = i - 1; dim0 >= 0; --dim0) {
        const int comparison = should_swap(perm[dim0], perm[dim1]);
        if (comparison > 0) {
          std::swap(perm[dim0], perm[dim1]);
          dim1 = dim0;
        } else if (comparison < 0) {
          break;
        }
      }
    }
    std::reverse(perm.begin(), perm.end());
  }
  std::vector<int64_t> result(ndim);
  int64_t expected = 1;
  for (int k = ndim - 1; k >= 0; --k) {
    result[perm[k]] = expected;
    expected *= std::max<int64_t>(sizes[perm[k]], 1);
  }
  return result;
}

template <typename T>
Error gather(
    const IndexPlan& plan,
    int64_t numel,
    const void* self,
    void* out,
    cudaStream_t stream) {
  gather_kernel<T><<<grid_size(numel), kThreads, 0, stream>>>(
      plan, numel, static_cast<const T*>(self), static_cast<T*>(out));
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  return Error::Ok;
}

template <typename T, typename AccT, bool kAccumulate>
Error put(
    const IndexPlan& plan,
    int64_t numel,
    bool may_repeat,
    void* out_data,
    const void* values_data,
    Scratch& scratch,
    cudaStream_t stream) {
  auto* out = static_cast<T*>(out_data);
  const auto* values = static_cast<const T*>(values_data);
  if (!may_repeat) {
    put_kernel<T, AccT, kAccumulate>
        <<<grid_size(numel), kThreads, 0, stream>>>(plan, numel, out, values);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
    return Error::Ok;
  }
  // One key per index point, so the scratch and the sort grow with the
  // number of indices, not with the elements they write.
  const int64_t points = plan.broadcast_numel;
  // CUB counts items in an int on CUDA 12.
  ET_CHECK_OR_RETURN_ERROR(
      points <= INT_MAX,
      NotSupported,
      "index_put: more than %d index points are not supported",
      INT_MAX);
  auto* offsets =
      static_cast<int64_t*>(scratch.allocate(4 * points * sizeof(int64_t)));
  ET_CHECK_OR_RETURN_ERROR(
      offsets != nullptr,
      MemoryAllocationFailed,
      "index_put: allocating the offsets failed");
  int64_t* out_offsets = offsets;
  int64_t* values_offsets = offsets + points;
  int64_t* sorted_out_offsets = offsets + 2 * points;
  int64_t* sorted_values_offsets = offsets + 3 * points;
  index_offsets_kernel<<<grid_size(points), kThreads, 0, stream>>>(
      plan, points, out_offsets, values_offsets);
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  const auto count = static_cast<int>(points);
  size_t temp_bytes = 0;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_bytes,
      out_offsets,
      sorted_out_offsets,
      values_offsets,
      sorted_values_offsets,
      count,
      0,
      static_cast<int>(sizeof(int64_t) * 8),
      stream));
  void* temp = scratch.allocate(temp_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      temp != nullptr,
      MemoryAllocationFailed,
      "index_put: allocating the sort scratch failed");
  ET_CUDA_CHECK_OR_RETURN_ERROR(cub::DeviceRadixSort::SortPairs(
      temp,
      temp_bytes,
      out_offsets,
      sorted_out_offsets,
      values_offsets,
      sorted_values_offsets,
      count,
      0,
      static_cast<int>(sizeof(int64_t) * 8),
      stream));
  put_sorted_kernel<T, AccT, kAccumulate>
      <<<grid_size(numel), kThreads, 0, stream>>>(
          plan,
          points,
          numel / points,
          sorted_out_offsets,
          sorted_values_offsets,
          out,
          values);
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  return Error::Ok;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_index_Tensor(
    Tensor* self,
    Tensor* const** indices,
    int64_t indices_len_,
    Tensor** ret0) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr && ret0 != nullptr &&
          (indices != nullptr || indices_len_ == 0),
      InvalidArgument,
      "aoti_torch_cuda_index_Tensor: null argument");
  ET_CHECK_OR_RETURN_ERROR(
      self->is_cuda(),
      NotSupported,
      "aoti_torch_cuda_index_Tensor: self is not on the CUDA device");
  const auto device = static_cast<int32_t>(self->device_index());
  auto stream_result = getCurrentCUDAStream(self->device_index());
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_cuda_index_Tensor: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  std::vector<Tensor*> index_tensors(indices_len_);
  for (int64_t i = 0; i < indices_len_; ++i) {
    index_tensors[i] = indices[i] == nullptr ? nullptr : *indices[i];
  }
  Scratch scratch(device, stream);
  IndexPlan plan;
  bool may_repeat = false;
  ET_CHECK_OK_OR_RETURN_ERROR(plan_indexing(
      *self,
      index_tensors.data(),
      indices_len_,
      scratch,
      stream,
      &plan,
      &may_repeat));

  const std::vector<int64_t> sizes(plan.sizes, plan.sizes + plan.ndim);
  const std::vector<int64_t> strides = output_strides(plan, self->numel() == 0);
  Tensor* result = nullptr;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_empty_strided(
      plan.ndim,
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(self->dtype()),
      static_cast<int32_t>(c10_slim::DeviceType::CUDA),
      device,
      &result));
  // The caller takes ownership of *ret0 only when this call succeeds.
  std::unique_ptr<Tensor, void (*)(Tensor*)> owner(result, [](Tensor* tensor) {
    (void)aoti_torch_delete_tensor_object(tensor);
  });
  std::copy(strides.begin(), strides.end(), plan.other_strides);

  const int64_t numel = plan_numel(plan);
  if (numel > 0) {
    const void* src = self->data_ptr();
    void* dst = result->data_ptr();
    switch (self->itemsize()) {
      case 1:
        ET_CHECK_OK_OR_RETURN_ERROR(
            gather<uint8_t>(plan, numel, src, dst, stream));
        break;
      case 2:
        ET_CHECK_OK_OR_RETURN_ERROR(
            gather<uint16_t>(plan, numel, src, dst, stream));
        break;
      case 4:
        ET_CHECK_OK_OR_RETURN_ERROR(
            gather<uint32_t>(plan, numel, src, dst, stream));
        break;
      case 8:
        ET_CHECK_OK_OR_RETURN_ERROR(
            gather<uint64_t>(plan, numel, src, dst, stream));
        break;
      default:
        ET_LOG(
            Error,
            "aoti_torch_cuda_index_Tensor: unsupported element size %zu",
            self->itemsize());
        return Error::NotSupported;
    }
  }
  *ret0 = owner.release();
  return Error::Ok;
}

AOTITorchError aoti_torch_index_put_out(
    Tensor* out,
    Tensor* self,
    Tensor* const* indices,
    const uint32_t num_indices,
    Tensor* values,
    bool accumulate) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr && values != nullptr &&
          (indices != nullptr || num_indices == 0),
      InvalidArgument,
      "aoti_torch_index_put_out: null argument");
  ET_CHECK_OR_RETURN_ERROR(
      out == self,
      NotSupported,
      "aoti_torch_index_put_out: out must be self, as AOTInductor calls it");
  ET_CHECK_OR_RETURN_ERROR(
      out->is_cuda() && values->is_cuda(),
      NotSupported,
      "aoti_torch_index_put_out: tensors must be on the CUDA device");
  ET_CHECK_OR_RETURN_ERROR(
      values->dtype() == out->dtype(),
      InvalidArgument,
      "aoti_torch_index_put_out: values and out must have the same dtype");
  const auto device = static_cast<int32_t>(out->device_index());
  auto stream_result = getCurrentCUDAStream(out->device_index());
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_index_put_out: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  Scratch scratch(device, stream);
  IndexPlan plan;
  bool may_repeat = false;
  ET_CHECK_OK_OR_RETURN_ERROR(plan_indexing(
      *out, indices, num_indices, scratch, stream, &plan, &may_repeat));

  const auto values_ndim = static_cast<int>(values->dim());
  ET_CHECK_OR_RETURN_ERROR(
      values_ndim <= plan.ndim,
      InvalidArgument,
      "aoti_torch_index_put_out: values cannot be broadcast to the indexed shape");
  const int lead = plan.ndim - values_ndim;
  for (int d = 0; d < values_ndim; ++d) {
    const int64_t size = values->size(d);
    ET_CHECK_OR_RETURN_ERROR(
        size == 1 || size == plan.sizes[lead + d],
        InvalidArgument,
        "aoti_torch_index_put_out: values cannot be broadcast to the indexed shape");
    plan.other_strides[lead + d] = size == 1 ? 0 : values->stride(d);
  }

  const int64_t numel = plan_numel(plan);
  if (numel == 0) {
    return Error::Ok;
  }
  void* dst = out->data_ptr();
  const void* src = values->data_ptr();
  if (!accumulate) {
    switch (out->itemsize()) {
      case 1:
        return put<uint8_t, uint8_t, false>(
            plan, numel, may_repeat, dst, src, scratch, stream);
      case 2:
        return put<uint16_t, uint16_t, false>(
            plan, numel, may_repeat, dst, src, scratch, stream);
      case 4:
        return put<uint32_t, uint32_t, false>(
            plan, numel, may_repeat, dst, src, scratch, stream);
      case 8:
        return put<uint64_t, uint64_t, false>(
            plan, numel, may_repeat, dst, src, scratch, stream);
      default:
        ET_LOG(
            Error,
            "aoti_torch_index_put_out: unsupported element size %zu",
            out->itemsize());
        return Error::NotSupported;
    }
  }
  // The accumulation types are ATen's.
  switch (out->dtype()) {
    case ScalarType::Float:
      return put<float, float, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Half:
      return put<__half, float, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::BFloat16:
      return put<__nv_bfloat16, float, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Long:
      return put<int64_t, int64_t, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Int:
      return put<int32_t, int64_t, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Short:
      return put<int16_t, int64_t, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Char:
      return put<int8_t, int64_t, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Byte:
      return put<uint8_t, int64_t, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    case ScalarType::Bool:
      return put<bool, bool, true>(
          plan, numel, may_repeat, dst, src, scratch, stream);
    default:
      ET_LOG(
          Error,
          "aoti_torch_index_put_out: unsupported dtype %d",
          static_cast<int>(out->dtype()));
      return Error::NotSupported;
  }
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
