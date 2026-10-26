// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the BSD-style license in LICENSE.

Error CudaKVPool::repartition_sequences(
    int columns,
    const std::vector<SequenceRange>& ranges,
    int64_t required,
    cudaStream_t stream,
    bool trim_unused,
    bool* committed) {
  if (committed) {
    *committed = false;
  }
  ET_CHECK_OR_RETURN_ERROR(
      columns >= 2 && columns <= declared_columns_ && required >= 0,
      InvalidArgument, "offgraph_kv: invalid independent sequence geometry");
  std::vector<int64_t> capacities(columns, 0), offsets(columns, 0);
  std::unordered_set<int> sources, destinations;
  for (const auto& range : ranges) {
    ET_CHECK_OR_RETURN_ERROR(
        range.to >= 0 && range.to < columns - 1 && range.capacity > 0 &&
            range.live >= 0 && range.live <= range.capacity &&
            destinations.insert(range.to).second &&
            (range.from == -1 ||
             (range.from >= 0 && range.from < columns_ - 1 &&
              sources.insert(range.from).second)),
        InvalidArgument, "offgraph_kv: invalid independent owner range");
    capacities[range.to] = range.capacity;
  }
  // Static padding writes only position zero into this private dummy range.
  capacities.back() = 1;
  for (int col = 1; col < columns; ++col) {
    offsets[col] = offsets[col - 1] + capacities[col - 1];
  }
  const int64_t total = offsets.back() + capacities.back();
  std::vector<int64_t> target_rows;
  bool changed = !independent_sequences_ || columns != columns_ ||
      capacities != sequence_capacities_;
  for (size_t index = 0; index < layers_.size(); ++index) {
    const auto& layer = layers_[index];
    const int64_t limit = layer.declared_rows / declared_columns_;
    for (const auto& range : ranges) {
      ET_CHECK_OR_RETURN_ERROR(
          layer.ring || range.capacity <= limit,
          InvalidArgument, "offgraph_kv: owner exceeds compiled context");
      changed |= range.from != range.to;
    }
    const int64_t previous = allocated_ ? allocations_[index].rows / columns_ :
        initial_rows_ / columns_;
    const int64_t ring_rows = std::min(limit,
        required > previous ? std::max(required, previous * 2) : previous);
    target_rows.push_back(layer.ring ? std::max<int64_t>(1, ring_rows) * columns : total);
    changed |= !allocated_ || target_rows.back() != allocations_[index].rows;
  }
  if (!changed) {
    const auto device = CUDAGuard::create(device_);
    ET_CHECK_OK_OR_RETURN_ERROR(device.error());
    ET_CHECK_OK_OR_RETURN_ERROR(follow_previous_step(stream));
    stream_ = stream;
    return Error::Ok;
  }
  const auto device = CUDAGuard::create(device_);
  ET_CHECK_OK_OR_RETURN_ERROR(device.error());
  ET_CHECK_OK_OR_RETURN_ERROR(synchronize(/*all_streams=*/true));
  stream_ = stream;
  if (!allocated_) {
    ET_CHECK_OK_OR_RETURN_ERROR(allocate_initial(0, stream));
  }
  const int old_columns = columns_;
  const bool old_independent = independent_sequences_;
  const auto old_capacities = sequence_capacities_;
  const auto old_offsets = sequence_offsets_;
  std::vector<int64_t> old_rows;
  for (const auto& allocation : allocations_) {
    old_rows.push_back(allocation.rows);
  }
  auto fence = [&]() -> Error {
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return Error::Internal;
    }
    if (trim_unused) {
      CudaAllocator::release_cached_memory(device_);
    }
    return Error::Ok;
  };
  auto copy = [&](size_t index, const Allocation& src,
                  Allocation& dst, bool reverse) -> Error {
    const auto& layer = layers_[index];
    const size_t bytes = row_bytes(layer);
    for (const auto& range : ranges) {
      if (range.from < 0) {
        continue;
      }
      const int old_col = range.from, new_col = range.to;
      const int src_col = reverse ? new_col : old_col;
      const int dst_col = reverse ? old_col : new_col;
      const int src_width = reverse ? columns : old_columns;
      const int dst_width = reverse ? old_columns : columns;
      const bool src_packed = !layer.ring && (reverse || old_independent);
      const bool dst_packed = !layer.ring && (!reverse || old_independent);
      const int64_t src_base = src_packed ?
          (reverse ? offsets[new_col] : old_offsets[old_col]) : src_col;
      const int64_t dst_base = dst_packed ?
          (reverse ? old_offsets[old_col] : offsets[new_col]) : dst_col;
      const int64_t rows = layer.ring ?
          std::min(src.rows / src_width, dst.rows / dst_width) : range.live;
      if (rows == 0) {
        continue;
      }
      for (const auto& pair : {std::pair{dst.k, src.k}, std::pair{dst.v, src.v}}) {
        const cudaError_t error = cudaMemcpy2DAsync(
            static_cast<uint8_t*>(pair.first) + bytes * dst_base,
            bytes * (dst_packed ? 1 : dst_width),
            static_cast<const uint8_t*>(pair.second) + bytes * src_base,
            bytes * (src_packed ? 1 : src_width), bytes,
            static_cast<size_t>(rows), cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) {
          return Error::Internal;
        }
      }
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? Error::Ok : Error::Internal;
  };
  std::vector<size_t> migrated;
  auto poison = [&]() -> Error {
    if (committed) {
      *committed = true;
    }
    invalidate_bindings();
    ET_LOG(Error, "offgraph_kv: independent owner migration cannot be restored");
    return Error::Internal;
  };
  auto rollback = [&](Error error) -> Error {
    if (fence() != Error::Ok) {
      return poison();
    }
    while (!migrated.empty()) {
      const size_t index = migrated.back();
      Allocation restored;
      if (allocate_layer(layers_[index], old_rows[index], stream, restored) != Error::Ok) {
        return poison();
      }
      if (copy(index, allocations_[index], restored, true) != Error::Ok) {
        discard(layers_[index], restored, stream);
        return poison();
      }
      discard(layers_[index], allocations_[index], stream);
      allocations_[index] = restored;
      migrated.pop_back();
      if (fence() != Error::Ok) {
        return poison();
      }
    }
    return error;
  };
  ET_CHECK_OK_OR_RETURN_ERROR(fence());
  invalidate_bindings();
  for (size_t index = 0; index < layers_.size(); ++index) {
    // A flat range can move at unchanged total size, so compare ownership too.
    bool needs_copy = target_rows[index] != allocations_[index].rows ||
        columns != old_columns || !old_independent ||
        (!layers_[index].ring && capacities != old_capacities);
    for (const auto& range : ranges) {
      needs_copy |= range.from != range.to;
    }
    if (!needs_copy) {
      continue;
    }
    Allocation next;
    const Error alloc_error = allocate_layer(layers_[index], target_rows[index], stream, next);
    if (alloc_error != Error::Ok) {
      return rollback(alloc_error);
    }
    const Error copy_error = copy(index, allocations_[index], next, false);
    if (copy_error != Error::Ok) {
      discard(layers_[index], next, stream);
      return rollback(copy_error);
    }
    discard(layers_[index], allocations_[index], stream);
    allocations_[index] = next;
    migrated.push_back(index);
    if (fence() != Error::Ok) {
      return poison();
    }
  }
  initial_rows_ = initial_rows_ / old_columns * columns;
  max_rows_ = max_rows_ / old_columns * columns;
  columns_ = columns;
  sequence_capacities_ = std::move(capacities);
  sequence_offsets_ = std::move(offsets);
  independent_sequences_ = true;
  rows_ = *std::max_element(sequence_capacities_.begin(), sequence_capacities_.end()) * columns;
  ++growth_count_;
  ++column_resize_count_;
  if (committed) {
    *committed = true;
  }
  ET_LOG(Info, "offgraph_kv: independent owners=%zu columns=%d flat_capacity_sum=%lld bytes=%lld",
      ranges.size(), columns, static_cast<long long>(total),
      static_cast<long long>(allocated_bytes_));
  return mark_step_done();
}
