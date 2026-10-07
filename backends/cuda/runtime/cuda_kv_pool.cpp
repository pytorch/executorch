/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>

#include <algorithm>
#include <limits>
#include <utility>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/backends/aoti/slim/util/size_util.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

namespace aoti = ::executorch::backends::aoti;
namespace slim = ::executorch::backends::aoti::slim;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::backends::aoti::slim::from_blob;
using ::executorch::backends::aoti::slim::SlimTensor;
using ::executorch::extension::cuda::CUDAGuard;
using ::executorch::runtime::Error;

std::string offgraph_kv_layer_fqn(int64_t layer_id, const char* suffix) {
  return std::string(kOffGraphKVFqnPrefix) + "layer_" +
      std::to_string(layer_id) + "_" + suffix;
}

namespace {

size_t contiguous_nbytes(
    const std::vector<int64_t>& sizes,
    slimc10::ScalarType dtype) {
  return slim::compute_storage_nbytes_contiguous(
      slim::makeArrayRef(sizes), slimc10::elementSize(dtype), 0);
}

auto shape_string(const std::vector<int64_t>& sizes) {
  return ::executorch::runtime::tensor_shape_to_c_string(
      ::executorch::runtime::Span<const int64_t>(sizes.data(), sizes.size()));
}

} // namespace

CudaKVPool::CudaKVPool(
    std::vector<Layer> layers,
    std::vector<SideBuffer> side_buffers,
    slimc10::ScalarType storage_dtype,
    int64_t initial_rows)
    : layers_(std::move(layers)),
      side_specs_(std::move(side_buffers)),
      storage_dtype_(storage_dtype),
      initial_rows_(initial_rows) {
  max_rows_ = std::numeric_limits<int64_t>::max();
  for (const Layer& layer : layers_) {
    if (layer.growable) {
      max_rows_ = std::min(max_rows_, layer.declared_rows);
    }
  }
}

// Nothing else can reach the pool once it is being destroyed. It never
// touches stream_: that is the caller's stream, which may already be gone.
// The device is drained instead, which covers any step still in flight on
// whatever stream ran it.
CudaKVPool::~CudaKVPool() {
  if (!device_known_) {
    return;
  }
  // A destructor cannot report, and the guard logs a failed restore itself.
  const auto device = CUDAGuard::create(device_);
  (void)device;
  (void)cudaDeviceSynchronize();
  for (auto& allocation : allocations_) {
    release(allocation.k, cudaStreamPerThread);
    release(allocation.v, cudaStreamPerThread);
  }
  for (void* buffer : side_buffers_) {
    release(buffer, cudaStreamPerThread);
  }
  (void)cudaStreamSynchronize(cudaStreamPerThread);
  if (last_step_done_ != nullptr) {
    (void)cudaEventDestroy(last_step_done_);
  }
}

runtime::Result<bool> CudaKVPool::note_handle(CudaDelegateHandle* handle) {
  if (!device_known_) {
    ET_CHECK_OR_RETURN_ERROR(
        cudaGetDevice(&device_) == cudaSuccess,
        Internal,
        "offgraph_kv: cannot query the current CUDA device");
    device_known_ = true;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(build_descriptors(handle));
  if (descriptors_[handle].empty()) {
    descriptors_.erase(handle);
    return false;
  }
  return true;
}

void CudaKVPool::forget_handle(CudaDelegateHandle* handle) {
  descriptors_.erase(handle);
  bound_.erase(handle);
}

bool CudaKVPool::serves(CudaDelegateHandle* handle) const {
  return descriptors_.find(handle) != descriptors_.end();
}

Error CudaKVPool::validate() const {
  for (size_t index = 0; index < layers_.size(); ++index) {
    for (const char* suffix : {"k", "v"}) {
      if (discovered_fqns_.count(offgraph_kv_layer_fqn(
              static_cast<int64_t>(index), suffix)) == 0) {
        ET_LOG(
            Error,
            "offgraph_kv: missing AOTI storage for layer %zu (%s)",
            index,
            suffix);
        return Error::InvalidProgram;
      }
    }
  }
  for (const SideBuffer& spec : side_specs_) {
    if (discovered_fqns_.count(spec.fqn) == 0) {
      ET_LOG(
          Error, "offgraph_kv: missing AOTI side buffer %s", spec.fqn.c_str());
      return Error::InvalidProgram;
    }
  }
  return Error::Ok;
}

Error CudaKVPool::prepare(
    int64_t required_rows,
    int64_t live_rows,
    cudaStream_t stream) {
  const auto device = CUDAGuard::create(device_);
  ET_CHECK_OK_OR_RETURN_ERROR(device.error());
  ET_CHECK_OK_OR_RETURN_ERROR(follow_previous_step(stream));
  stream_ = stream;
  if (!allocated_) {
    ET_CHECK_OK_OR_RETURN_ERROR(allocate_initial(required_rows, stream));
  }
  if (required_rows > rows_) {
    const int64_t next = std::min<int64_t>(
        max_rows_, std::max<int64_t>(required_rows, rows_ * 2));
    ET_CHECK_OK_OR_RETURN_ERROR(grow(next, live_rows, stream));
  }
  return Error::Ok;
}

Error CudaKVPool::mark_step_done() {
  const auto device = CUDAGuard::create(device_);
  ET_CHECK_OK_OR_RETURN_ERROR(device.error());
  if (last_step_done_ == nullptr) {
    const cudaError_t error =
        cudaEventCreateWithFlags(&last_step_done_, cudaEventDisableTiming);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        Internal,
        "offgraph_kv: cannot create the step event: %s",
        cudaGetErrorString(error));
  }
  const cudaError_t error = cudaEventRecord(last_step_done_, stream_);
  ET_CHECK_OR_RETURN_ERROR(
      error == cudaSuccess,
      Internal,
      "offgraph_kv: cannot record the step event: %s",
      cudaGetErrorString(error));
  return Error::Ok;
}

void* CudaKVPool::side_buffer(size_t index) const {
  return index < side_buffers_.size() ? side_buffers_[index] : nullptr;
}

size_t CudaKVPool::side_buffer_bytes(size_t index) const {
  const SideBuffer& spec = side_specs_.at(index);
  return contiguous_nbytes(spec.sizes, spec.dtype);
}

size_t CudaKVPool::row_bytes(const Layer& layer) const {
  return static_cast<size_t>(layer.n_kv_heads) *
      static_cast<size_t>(layer.head_dim) *
      slimc10::elementSize(storage_dtype_);
}

// BSHD at the declared rows: no stride depends on the rows, so the storage
// behind it may hold fewer.
std::vector<int64_t> CudaKVPool::layer_sizes(const Layer& layer) {
  return {1, layer.declared_rows, layer.n_kv_heads, layer.head_dim};
}

Error CudaKVPool::allocate_layer(
    const Layer& layer,
    int64_t rows,
    cudaStream_t stream,
    Allocation& out) {
  const size_t bytes = row_bytes(layer) * static_cast<size_t>(rows);
  auto k = CudaAllocator::allocate_async(bytes, device_, stream);
  ET_CHECK_OK_OR_RETURN_ERROR(k.error());
  auto v = CudaAllocator::allocate_async(bytes, device_, stream);
  if (!v.ok()) {
    CudaAllocator::deallocate_async(k.get(), device_, stream);
    return v.error();
  }
  out = Allocation{k.get(), v.get(), rows};
  allocated_bytes_ += static_cast<int64_t>(2 * bytes);
  return Error::Ok;
}

void CudaKVPool::release(void* ptr, cudaStream_t stream) {
  if (device_known_ && ptr != nullptr) {
    CudaAllocator::deallocate_async(ptr, device_, stream);
  }
}

void CudaKVPool::discard(
    const Layer& layer,
    Allocation& allocation,
    cudaStream_t stream) {
  allocated_bytes_ -=
      static_cast<int64_t>(2 * row_bytes(layer) * allocation.rows);
  release(allocation.k, stream);
  release(allocation.v, stream);
  allocation = Allocation{};
}

// The previous step may have run on another stream (a caller-selected one),
// and its kernels may still be writing the storage this step reads, or that a
// growth below copies and frees. Order this stream behind it.
Error CudaKVPool::follow_previous_step(cudaStream_t stream) {
  if (last_step_done_ == nullptr || stream == stream_) {
    return Error::Ok;
  }
  const cudaError_t error = cudaStreamWaitEvent(stream, last_step_done_, 0);
  ET_CHECK_OR_RETURN_ERROR(
      error == cudaSuccess,
      Internal,
      "offgraph_kv: cannot order the step stream behind the previous one: %s",
      cudaGetErrorString(error));
  return Error::Ok;
}

// A first step wider than the initial rows is allocated at its own width
// rather than allocated and immediately grown. All or nothing: a failure frees
// what was allocated so far, and the next step may try again.
Error CudaKVPool::allocate_initial(int64_t required_rows, cudaStream_t stream) {
  const int64_t growable_rows = std::max<int64_t>(initial_rows_, required_rows);
  std::vector<Allocation> allocated;
  allocated.reserve(layers_.size());
  auto roll_back = [&]() {
    for (size_t index = 0; index < allocated.size(); ++index) {
      discard(layers_[index], allocated[index], stream);
    }
  };
  for (const Layer& layer : layers_) {
    const int64_t rows = layer.growable ? growable_rows : layer.declared_rows;
    Allocation allocation;
    const Error error = allocate_layer(layer, rows, stream, allocation);
    if (error != Error::Ok) {
      roll_back();
      return error;
    }
    allocated.push_back(allocation);
  }
  const Error side_error = allocate_side_buffers(stream);
  if (side_error != Error::Ok) {
    roll_back();
    return side_error;
  }
  allocations_ = std::move(allocated);
  rows_ = growable_rows;
  allocated_ = true;
  ET_LOG(
      Info,
      "offgraph_kv: initialized flat_capacity=%lld allocated_bytes=%lld",
      static_cast<long long>(rows_),
      static_cast<long long>(allocated_bytes_));
  return Error::Ok;
}

// Side buffers start zeroed, so a program bound before its first write reads
// a well-defined value rather than whatever the allocator handed back.
Error CudaKVPool::allocate_side_buffers(cudaStream_t stream) {
  std::vector<void*> buffers;
  buffers.reserve(side_specs_.size());
  auto roll_back = [&]() {
    for (void* buffer : buffers) {
      release(buffer, stream);
    }
  };
  for (size_t index = 0; index < side_specs_.size(); ++index) {
    const size_t bytes = side_buffer_bytes(index);
    auto buffer = CudaAllocator::allocate_async(bytes, device_, stream);
    if (!buffer.ok()) {
      roll_back();
      return buffer.error();
    }
    buffers.push_back(buffer.get());
    const cudaError_t error = cudaMemsetAsync(buffer.get(), 0, bytes, stream);
    if (error != cudaSuccess) {
      roll_back();
      ET_LOG(
          Error,
          "offgraph_kv: cannot clear side buffer %s: %s",
          side_specs_[index].fqn.c_str(),
          cudaGetErrorString(error));
      return Error::Internal;
    }
  }
  side_buffers_ = std::move(buffers);
  return Error::Ok;
}

// Reallocates every growable layer at new_rows and carries the rows already
// written across. BSHD storage makes that one contiguous prefix per buffer.
// Everything is ordered on `stream`, so the copy runs after the last step
// that wrote the old storage, the old storage is freed only after the copy,
// and the next step's kernels see the copied rows.
//
// Transactional: every replacement is allocated and filled before any old
// storage is released. A failure on any layer frees the replacements and
// leaves the pool exactly as it was -- storage, bindings, rows -- so the step
// fails but the pool stays usable.
Error CudaKVPool::grow(
    int64_t new_rows,
    int64_t live_rows,
    cudaStream_t stream) {
  const int64_t old_rows = rows_;
  std::vector<std::pair<size_t, Allocation>> replacements;
  auto roll_back = [&]() {
    for (auto& [index, replacement] : replacements) {
      discard(layers_[index], replacement, stream);
    }
  };
  for (size_t index = 0; index < layers_.size(); ++index) {
    const Layer& layer = layers_[index];
    if (!layer.growable) {
      continue;
    }
    Allocation replacement;
    const Error error = allocate_layer(layer, new_rows, stream, replacement);
    if (error != Error::Ok) {
      roll_back();
      return error;
    }
    replacements.emplace_back(index, replacement);
  }
  for (const auto& [index, replacement] : replacements) {
    const Allocation& current = allocations_.at(index);
    const size_t live_bytes =
        row_bytes(layers_[index]) * static_cast<size_t>(live_rows);
    if (live_bytes == 0) {
      continue;
    }
    for (const auto& [dst, src] :
         {std::pair{replacement.k, current.k},
          std::pair{replacement.v, current.v}}) {
      const cudaError_t copy_error = cudaMemcpyAsync(
          dst, src, live_bytes, cudaMemcpyDeviceToDevice, stream);
      if (copy_error != cudaSuccess) {
        ET_LOG(
            Error,
            "offgraph_kv: growth copy failed: %s",
            cudaGetErrorString(copy_error));
        roll_back();
        return Error::Internal;
      }
    }
  }
  for (auto& [index, replacement] : replacements) {
    Allocation& current = allocations_.at(index);
    discard(layers_[index], current, stream);
    current = replacement;
  }
  rows_ = new_rows;
  growth_count_++;
  // Every program sharing this pool now points at freed storage: drop the
  // bindings so each rebinds before its next run, and any captured CUDA graph
  // so it is captured again against the new storage. Prefill usually grows
  // the cache while decode's graph sits idle, so this reaches every handle,
  // not only the one stepping now.
  //
  // Rebinding also resets AOTI's constant-fold state, which must be run
  // eagerly, so every graph-enabled handle gets at least one eager step
  // before it captures -- including one that was about to capture for the
  // first time, and without shortening a longer warmup still outstanding.
  bound_.clear();
  for (auto& entry : descriptors_) {
    CudaGraphState& graph = entry.first->cuda_graph_state;
    if (graph.phase == CudaGraphPhase::Replay) {
      graph.recapture();
    } else if (graph.phase == CudaGraphPhase::Warmup) {
      graph.warmup_remaining = std::max(graph.warmup_remaining, 1);
    }
  }
  ET_LOG(
      Info,
      "offgraph_kv: grew flat_capacity=%lld->%lld allocated_bytes=%lld "
      "growth_count=%lld",
      static_cast<long long>(old_rows),
      static_cast<long long>(new_rows),
      static_cast<long long>(allocated_bytes_),
      static_cast<long long>(growth_count_));
  return Error::Ok;
}

Error CudaKVPool::build_descriptors(CudaDelegateHandle* handle) {
  ET_CHECK_OR_RETURN_ERROR(
      handle->get_num_constants && handle->get_constant_name &&
          handle->get_constant_original_fqn &&
          handle->update_user_managed_constant_buffer_pairs,
      NotSupported,
      "offgraph_kv: AOTI external-buffer APIs are unavailable");
  size_t count = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      handle->get_num_constants(handle->container_handle, &count));
  struct Compiled {
    std::string internal_name;
    size_t index;
  };
  std::unordered_map<std::string, Compiled> compiled;
  for (size_t index = 0; index < count; ++index) {
    const char* internal = nullptr;
    const char* original = nullptr;
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->get_constant_name(handle->container_handle, index, &internal));
    ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_original_fqn(
        handle->container_handle, index, &original));
    if (internal && original && internal[0] && original[0]) {
      compiled.emplace(original, Compiled{internal, index});
    }
  }

  std::vector<Descriptor> descriptors;
  std::vector<std::string> found_fqns;
  size_t found_layers = 0;
  for (size_t index = 0; index < layers_.size(); ++index) {
    size_t found = 0;
    for (const auto& [suffix, slot] :
         {std::pair{"k", Slot::Key}, std::pair{"v", Slot::Value}}) {
      const std::string name =
          offgraph_kv_layer_fqn(static_cast<int64_t>(index), suffix);
      const auto it = compiled.find(name);
      if (it == compiled.end()) {
        continue;
      }
      ET_CHECK_OK_OR_RETURN_ERROR(check_compiled(
          handle,
          it->second.index,
          name,
          storage_dtype_,
          layer_sizes(layers_[index])));
      ++found;
      descriptors.push_back(Descriptor{it->second.internal_name, slot, index});
      found_fqns.push_back(name);
    }
    if (found == 2) {
      ++found_layers;
    }
  }
  size_t found_side = 0;
  for (size_t index = 0; index < side_specs_.size(); ++index) {
    const SideBuffer& spec = side_specs_[index];
    const auto it = compiled.find(spec.fqn);
    if (it != compiled.end()) {
      ET_CHECK_OK_OR_RETURN_ERROR(check_compiled(
          handle, it->second.index, spec.fqn, spec.dtype, spec.sizes));
      ++found_side;
      descriptors.push_back(
          Descriptor{it->second.internal_name, Slot::Side, index});
      found_fqns.push_back(spec.fqn);
    }
  }
  // A method either has no off-graph storage (embeddings, vision) or has all
  // of it. Anything between means the lowering pass and this runtime disagree
  // about the geometry, which is worth failing on here -- while the offending
  // method is still named -- rather than at the first step.
  ET_CHECK_OR_RETURN_ERROR(
      found_layers == 0 || found_layers == layers_.size(),
      InvalidProgram,
      "offgraph_kv: program carries %zu of %zu layers' storage",
      found_layers,
      layers_.size());
  ET_CHECK_OR_RETURN_ERROR(
      found_side == (found_layers == 0 ? 0 : side_specs_.size()),
      InvalidProgram,
      "offgraph_kv: program carries %zu of %zu side buffers",
      found_side,
      side_specs_.size());
  discovered_fqns_.insert(found_fqns.begin(), found_fqns.end());
  // A reload of the same handle replaces what it had rather than adding to
  // it, so its constants are never bound twice.
  descriptors_[handle] = std::move(descriptors);
  bound_.erase(handle);
  return Error::Ok;
}

// The program's kernels address its storage with the shape and dtype it was
// compiled with, and AOTI binds external buffers without checking either. So
// the pool's idea of each constant -- dtype, and shape (BSHD at the declared
// rows for a layer, the declared shape for a side buffer) -- must match what
// the program declared exactly, or a step could index past the allocation, or
// wrap a ring at another window than the cache does.
//
// AOTI exposes no constant shapes, so the shape comes from the FQN-weight
// metadata serialized alongside the program. The bytes AOTI reports only
// cross-check that metadata against the library: they are rounded up to a
// multiple of 64 when the program also holds CPU constants
// (cpp_wrapper_cpu.py), so they cannot tell nearby shapes apart on their own.
Error CudaKVPool::check_compiled(
    CudaDelegateHandle* handle,
    size_t constant_index,
    const std::string& name,
    slimc10::ScalarType dtype,
    const std::vector<int64_t>& sizes) const {
  ET_CHECK_OR_RETURN_ERROR(
      handle->get_constant_dtype && handle->get_constant_data_size,
      NotSupported,
      "offgraph_kv: AOTI constant metadata APIs are unavailable");
  const auto compiled_sizes = handle->offgraph_kv_sizes.find(name);
  ET_CHECK_OR_RETURN_ERROR(
      compiled_sizes != handle->offgraph_kv_sizes.end(),
      InvalidProgram,
      "offgraph_kv: %s has no compiled shape in the program's FQN-weight "
      "metadata",
      name.c_str());
  ET_CHECK_OR_RETURN_ERROR(
      compiled_sizes->second == sizes,
      InvalidProgram,
      "offgraph_kv: %s is compiled with shape %s but the cache declares %s; "
      "the cache's capacity, max_write, window or head geometry does not "
      "match the program",
      name.c_str(),
      shape_string(compiled_sizes->second).data(),
      shape_string(sizes).data());
  int32_t compiled_dtype = 0;
  size_t compiled_bytes = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_dtype(
      handle->container_handle, constant_index, &compiled_dtype));
  ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_data_size(
      handle->container_handle, constant_index, &compiled_bytes));
  ET_CHECK_OR_RETURN_ERROR(
      compiled_dtype == static_cast<int32_t>(dtype),
      InvalidProgram,
      "offgraph_kv: %s is compiled as dtype %d but the cache stores %d",
      name.c_str(),
      static_cast<int>(compiled_dtype),
      static_cast<int>(dtype));
  constexpr size_t kAotiConstantAlignment = 64;
  const size_t bytes = contiguous_nbytes(sizes, dtype);
  const size_t aligned = (bytes + kAotiConstantAlignment - 1) /
      kAotiConstantAlignment * kAotiConstantAlignment;
  ET_CHECK_OR_RETURN_ERROR(
      compiled_bytes == bytes || compiled_bytes == aligned,
      InvalidProgram,
      "offgraph_kv: %s is compiled with %zu bytes but its serialized shape "
      "spans %zu; the program's metadata does not match its library",
      name.c_str(),
      compiled_bytes,
      bytes);
  return Error::Ok;
}

// Binds each constant with the shape the program declared over the current
// allocation. A growable layer is declared BSHD at its maximum rows but may be
// backed by fewer: every access the program makes is bounded by kv_len along
// the sequence, and prepare() has grown the allocation past it.
Error CudaKVPool::bind(CudaDelegateHandle* handle) {
  if (bound_.find(handle) != bound_.end()) {
    return Error::Ok;
  }
  const auto descriptors = descriptors_.find(handle);
  if (descriptors == descriptors_.end()) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      allocated_, InvalidState, "offgraph_kv: prepare must run before bind");
  Bound bound;
  bound.tensors.reserve(descriptors->second.size());
  std::vector<aoti::AOTInductorConstantMapEntry> pairs;
  pairs.reserve(descriptors->second.size());
  const slimc10::Device device(slimc10::DeviceType::CUDA, device_);
  for (const Descriptor& descriptor : descriptors->second) {
    void* data = nullptr;
    std::vector<int64_t> sizes;
    slimc10::ScalarType dtype = storage_dtype_;
    if (descriptor.slot == Slot::Side) {
      const SideBuffer& spec = side_specs_[descriptor.index];
      data = side_buffers_[descriptor.index];
      sizes = spec.sizes;
      dtype = spec.dtype;
    } else {
      const Layer& layer = layers_[descriptor.index];
      const Allocation& allocation = allocations_[descriptor.index];
      ET_CHECK_OR_RETURN_ERROR(
          allocation.rows <= layer.declared_rows,
          Internal,
          "offgraph_kv: layer %zu holds %lld rows, more than the %lld declared",
          descriptor.index,
          static_cast<long long>(allocation.rows),
          static_cast<long long>(layer.declared_rows));
      data = descriptor.slot == Slot::Value ? allocation.v : allocation.k;
      sizes = layer_sizes(layer);
    }
    auto tensor = std::make_unique<SlimTensor>(
        from_blob(data, slim::makeArrayRef(sizes), dtype, device));
    pairs.push_back(
        {descriptor.internal_name.c_str(),
         reinterpret_cast<aoti::AtenTensorHandle>(tensor.get())});
    bound.tensors.push_back(std::move(tensor));
  }
  if (!pairs.empty()) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->update_user_managed_constant_buffer_pairs(
            handle->container_handle,
            pairs.data(),
            pairs.size(),
            /*use_inactive=*/false,
            /*validate_full_update=*/false));
  }
  bound_.emplace(handle, std::move(bound));
  return Error::Ok;
}

} // namespace executorch::backends::cuda
