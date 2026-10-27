// Copyright (c) Meta Platforms, Inc. and affiliates.
// BSD-style license in LICENSE. Included inside CudaBatchedSequenceCache.
// A host undo archive preserves the existing max_write rewind contract while
// only window rows per owner remain on the device. The packed GPU staging
// buffer is shared across owners; one pitched transfer copies real tokens.
struct RingRun {
  int32_t owner;
  int packed_start;
  int length;
  int absolute_start;
};
std::vector<RingRun> ring_runs_;
std::vector<RingRun> journal_runs_;
std::vector<size_t> ring_layers_;
static constexpr int kHostRingPageRows = 64;
using HostRingPage = std::shared_ptr<std::vector<uint8_t>>;
// Prefix clones share immutable host pages. A write copies just the affected
// page, rather than copying a window+max_write archive at every prefix hit.
std::map<int32_t, std::vector<HostRingPage>> ring_archives_;
void* ring_host_step_ = nullptr;
cudaEvent_t ring_copy_done_ = nullptr;
bool ring_copy_pending_ = false;
int ring_step_tokens_ = 0;
int ring_journal_tokens_ = 0;
size_t ring_row_bytes_ = 0;
int64_t ring_window_ = 0;
int64_t ring_archive_rows_ = 0;
uint64_t ring_D2H_bytes_ = 0;
uint64_t ring_restore_bytes_ = 0;

Error initialize_ring_journal() {
  if (!cfg_.split_ring_history) return Error::Ok;
  for (size_t i = 0; i < geometry_.layers.size(); ++i) {
    const auto& l = geometry_.layers[i];
    if (!is_ring(l)) continue;
    const size_t bytes = static_cast<size_t>(l.n_kv_heads) * l.head_dim * 2;
    if (ring_layers_.empty()) {
      ring_row_bytes_ = bytes;
      ring_window_ = l.policy.window;
      ring_archive_rows_ = std::min<int64_t>(cfg_.capacity, ring_window_ + max_write_ - 1);
    } else {
      ET_CHECK_OR_RETURN_ERROR(bytes == ring_row_bytes_ && l.policy.window == ring_window_,
          NotSupported, "split ring journal requires uniform ring geometry");
    }
    ring_layers_.push_back(i);
  }
  ET_CHECK_OR_RETURN_ERROR(!ring_layers_.empty(), InvalidArgument, "split ring needs ring layers");
  const size_t host_bytes = ring_layers_.size() * 2 * max_write_ * ring_row_bytes_;
  ET_CHECK_OR_RETURN_ERROR(cudaHostAlloc(&ring_host_step_, host_bytes, cudaHostAllocDefault) == cudaSuccess,
      MemoryAllocationFailed, "cannot allocate split ring host staging");
  ET_CHECK_OR_RETURN_ERROR(cudaEventCreateWithFlags(&ring_copy_done_, cudaEventDisableTiming) == cudaSuccess,
      Internal, "cannot create split ring copy event");
  return Error::Ok;
}

Error finish_ring_journal() {
  if (!ring_copy_pending_) return Error::Ok;
  ET_CHECK_OR_RETURN_ERROR(cudaEventSynchronize(ring_copy_done_) == cudaSuccess,
      Internal, "split ring journal transfer failed");
  const auto* step = static_cast<const uint8_t*>(ring_host_step_);
  const int64_t pages = (ring_archive_rows_ + kHostRingPageRows - 1) / kHostRingPageRows;
  const size_t page_plane_bytes = kHostRingPageRows * ring_row_bytes_;
  for (const auto& run : journal_runs_) {
    auto& archive = ring_archives_[run.owner];
    if (archive.empty()) archive.resize(pages);
    int done = 0;
    while (done < run.length) {
      const int64_t dst = (run.absolute_start + done) % ring_archive_rows_;
      const int64_t block = dst / kHostRingPageRows;
      const int64_t row = dst % kHostRingPageRows;
      const int n = std::min<int64_t>(run.length - done,
          std::min(ring_archive_rows_ - dst, kHostRingPageRows - row));
      auto& page = archive[block];
      if (!page) page = std::make_shared<std::vector<uint8_t>>(ring_layers_.size() * 2 * page_plane_bytes);
      else if (!page.unique()) page = std::make_shared<std::vector<uint8_t>>(*page);
      for (size_t plane = 0; plane < ring_layers_.size() * 2; ++plane) {
        std::memcpy(page->data() + plane * page_plane_bytes + row * ring_row_bytes_,
            step + (plane * ring_journal_tokens_ + run.packed_start + done) * ring_row_bytes_,
            static_cast<size_t>(n) * ring_row_bytes_);
      }
      done += n;
    }
  }
  journal_runs_.clear();
  ring_copy_pending_ = false;
  return Error::Ok;
}

Error copy_ring_journal() {
  if (!cfg_.split_ring_history) return Error::Ok;
  ET_CHECK_OR_RETURN_ERROR(!ring_copy_pending_, InvalidState, "split ring staging still pending");
  const size_t width = static_cast<size_t>(ring_step_tokens_) * ring_row_bytes_;
  ET_CHECK_OR_RETURN_ERROR(cudaMemcpy2DAsync(ring_host_step_, width,
      pool_.side_buffer(3), static_cast<size_t>(max_write_) * ring_row_bytes_,
      width, ring_layers_.size() * 2, cudaMemcpyDeviceToHost, pool_.stream()) == cudaSuccess,
      Internal, "cannot copy split ring packed staging");
  ET_CHECK_OR_RETURN_ERROR(cudaEventRecord(ring_copy_done_, pool_.stream()) == cudaSuccess,
      Internal, "cannot fence split ring staging");
  ring_D2H_bytes_ += width * ring_layers_.size() * 2;
  ring_journal_tokens_ = ring_step_tokens_;
  journal_runs_ = std::move(ring_runs_);
  ring_copy_pending_ = true;
  return Error::Ok;
}

Error restore_ring_history(int32_t owner, int column, int keep) {
  if (!cfg_.split_ring_history || keep == 0) return Error::Ok;
  ET_CHECK_OK_OR_RETURN_ERROR(finish_ring_journal());
  auto it = ring_archives_.find(owner);
  ET_CHECK_OR_RETURN_ERROR(it != ring_archives_.end(), InvalidState, "split ring rewind archive missing");
  const int first = std::max<int64_t>(0, keep - ring_window_);
  const size_t page_plane_bytes = kHostRingPageRows * ring_row_bytes_;
  for (size_t r = 0; r < ring_layers_.size(); ++r) {
    for (int v = 0; v < 2; ++v) {
      int pos = first;
      while (pos < keep) {
        const int64_t hr = pos % ring_archive_rows_;
        const int64_t dr = pos % ring_window_;
        const int64_t page_row = hr % kHostRingPageRows;
        const int n = std::min<int64_t>(keep - pos,
            std::min(kHostRingPageRows - page_row, std::min(ring_archive_rows_ - hr, ring_window_ - dr)));
        const auto& page = it->second.at(hr / kHostRingPageRows);
        ET_CHECK_OR_RETURN_ERROR(page != nullptr, InvalidState, "split ring rewind host page missing");
        const auto* host = page->data() + (r * 2 + v) * page_plane_bytes + page_row * ring_row_bytes_;
        auto* device = static_cast<uint8_t*>(pool_.layer_data(ring_layers_[r], v != 0)) +
            (dr * columns_ + column) * ring_row_bytes_;
        ET_CHECK_OR_RETURN_ERROR(cudaMemcpy2DAsync(device, columns_ * ring_row_bytes_,
            host, ring_row_bytes_, ring_row_bytes_, n, cudaMemcpyHostToDevice, pool_.stream()) == cudaSuccess,
            Internal, "cannot restore split ring rewind");
        ring_restore_bytes_ += static_cast<uint64_t>(n) * ring_row_bytes_;
        pos += n;
      }
    }
  }
  // Pageable undo pages may be freed or copied on a later owner operation.
  // Complete this infrequent restore before releasing their host lifetime.
  ET_CHECK_OR_RETURN_ERROR(cudaStreamSynchronize(pool_.stream()) == cudaSuccess,
      Internal, "cannot complete split ring rewind restore");
  return Error::Ok;
}

void clear_ring_journal() {
  if (ring_copy_pending_ && finish_ring_journal() != Error::Ok) poisoned_ = true;
  ring_runs_.clear(); journal_runs_.clear(); ring_archives_.clear();
}
void destroy_ring_journal() {
  if (ring_copy_pending_) (void)finish_ring_journal();
  if (ring_copy_done_) (void)cudaEventDestroy(ring_copy_done_);
  if (ring_host_step_) (void)cudaFreeHost(ring_host_step_);
}
