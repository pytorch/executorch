/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "MLXCache.h"

#include <executorch/extension/llm/cache/cache.h>

namespace executorch::backends::mlx {

class MLXSequenceCacheV2;

namespace cache = ::executorch::extension::llm::cache;
namespace sequence_v2 {

struct KV {
  Tensor k;
  Tensor v;
};

namespace detail {

class KVAttentionLayer;

inline void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

inline bool floating(::mlx::core::Dtype dtype) {
  return dtype == ::mlx::core::float16 || dtype == ::mlx::core::bfloat16 ||
      dtype == ::mlx::core::float32;
}

inline Tensor
cast_if_needed(const Tensor& t, ::mlx::core::Dtype dtype, StreamOrDevice s) {
  return t.dtype() == dtype ? t : ::mlx::core::astype(t, dtype, s);
}

inline int count(const std::optional<KV>& kv) {
  return kv ? kv->k.shape(2) : 0;
}

inline Tensor tokens(const Tensor& t, int start, int length, StreamOrDevice s) {
  return ::mlx::core::slice(
      t, {0, 0, start, 0}, {1, t.shape(1), start + length, t.shape(3)}, s);
}

struct Segments {
  std::optional<KV> head;
  std::optional<KV> tail;
};

// One cache-owned mask, shared across layers but never owned by a policy.
struct CachedMask {
  int T, S, window;
  Tensor mask;
  ::mlx::core::Stream stream;
};

} // namespace detail

// Bound attention behavior and mutable per-sequence state. KV is the raw input
// pair, not a constraint on the layer's persistent representation.
class AttentionLayer {
 public:
  struct QuerySpan {
    int begin;
    int count;
  };
  struct UpdateInfo {
    int end;
    bool overwrite;
  };

  virtual ~AttentionLayer() = default;
  virtual int length() const = 0;
  virtual QuerySpan query_span(int kv_tokens) const = 0;
  virtual std::optional<int> admission(int position, int kv_tokens) const = 0;
  // n is logical persistent growth, not a raw input K/V count.
  virtual bool can_extend(int n) const = 0;
  virtual bool can_rewind(int position) const = 0;
  // Zero must reset even unfinished state; other targets are preflighted.
  virtual void rewind(int position) = 0;
  virtual void clear() = 0;
  virtual std::unique_ptr<AttentionLayer> clone_current() const = 0;
  virtual UpdateInfo validate(
      int position,
      int complete_prefix,
      const Tensor& q,
      const KV& incoming,
      float scale) const = 0;
  // Transactional: failure preserves this layer's state; success publishes the
  // validated logical end (or preserves the later suffix for an overwrite).
  virtual Tensor update_and_attend(
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      const UpdateInfo& update,
      StreamOrDevice s,
      std::optional<detail::CachedMask>& mask) = 0;
};

// Immutable shared factory for attention layers. Invoke the hook on the owner
// passed to it so implementations can safely retain typed factory ownership.
class AttentionLayerFactory {
 public:
  virtual ~AttentionLayerFactory() = default;

 private:
  friend class ::executorch::backends::mlx::MLXSequenceCacheV2;

  static std::unique_ptr<AttentionLayer> create(
      std::shared_ptr<const AttentionLayerFactory> owner,
      const cache::LayerGeometry& geometry,
      const cache::CacheConfig& cfg) {
    detail::require(owner != nullptr, "sequence v2: null layer factory");
    auto layer = owner->create_layer(owner, geometry, cfg);
    detail::require(layer != nullptr, "sequence v2: null attention layer");
    detail::require(
        layer->length() == 0, "sequence v2: initial layer must be empty");
    return layer;
  }

  virtual std::unique_ptr<AttentionLayer> create_layer(
      const std::shared_ptr<const AttentionLayerFactory>& owner,
      const cache::LayerGeometry& geometry,
      const cache::CacheConfig& cfg) const = 0;
};

// Integer layout and storage only: attention intervals and masks belong to
// policies. Copies share functional arrays during graph construction. Publish
// the new roots before caller-side execution so old owners do not block
// donation.
class Storage {
 public:
  Storage(const cache::LayerGeometry& geometry, const cache::CacheConfig& cfg)
      : capacity_(cfg.capacity),
        ring_(geometry.policy.kind == cache::LayerPolicy::Kind::Ring),
        heads_(geometry.n_kv_heads),
        head_dim_(geometry.head_dim),
        dtype_(::mlx::core::float32) {
    detail::require(
        cache::valid(cfg) && heads_ > 0 && head_dim_ > 0 &&
            ((ring_ && geometry.policy.window > 0) ||
             (geometry.policy.kind == cache::LayerPolicy::Kind::Flat &&
              geometry.policy.window == 0)),
        "storage: invalid geometry/config");
    detail::require(
        cfg.kv_dtype == static_cast<int>(ScalarType::Half) ||
            cfg.kv_dtype == static_cast<int>(ScalarType::BFloat16) ||
            cfg.kv_dtype == static_cast<int>(ScalarType::Float),
        "storage: dtype must be float16, bfloat16 or float32");
    dtype_ = resolve_dtype(static_cast<int8_t>(cfg.kv_dtype));
    max_write_ =
        cfg.max_write.value_or(ring_ ? geometry.policy.window : capacity_);
    const int64_t slots =
        ring_ ? int64_t{geometry.policy.window} + max_write_ - 1 : capacity_;
    detail::require(
        slots > 0 && slots <= std::numeric_limits<int>::max(),
        "storage: physical storage exceeds integer bounds");
    slots_ = static_cast<int>(slots);
    initial_slots_ = ring_ ? slots_ : std::min(cfg.initial_capacity, slots_);
    check_allocation_bounded(
        {1, heads_, slots_, head_dim_}, dtype_, "sequence v2 storage");
  }

  int capacity() const {
    return capacity_;
  }
  int length() const {
    return length_;
  }
  int highwater() const {
    return highwater_;
  }
  int max_write() const {
    return max_write_;
  }
  int slots() const {
    return slots_;
  }
  int heads() const {
    return heads_;
  }
  int head_dim() const {
    return head_dim_;
  }

  bool retains(int begin, int end) const {
    const int oldest = ring_ ? std::max(0, highwater_ - slots_) : 0;
    return begin >= oldest && begin <= end && end <= length_;
  }

  bool can_rewind(int position, int required_begin) const {
    return position == 0 ||
        (position > 0 && position <= length_ &&
         retains(required_begin, position));
  }

  void rewind(int position, int required_begin) {
    detail::require(
        can_rewind(position, required_begin), "storage: unavailable rewind");
    length_ = position;
    // Only a full logical reset can forget the previous eviction boundary.
    if (position == 0) {
      highwater_ = 0;
    }
  }

  void clear() {
    buffers_.reset();
    length_ = highwater_ = 0;
  }

  // Cache has already validated tensor ranks, geometry, dtypes and paired
  // lengths. Check storage invariants, then write every supplied K/V token.
  // Updates may overlap retained history but never truncate its later suffix.
  void update(int position, const std::optional<KV>& kv, StreamOrDevice s) {
    const int n = detail::count(kv);
    detail::require(
        position >= 0 && position <= length_ && n >= 0 && n <= max_write_ &&
            n <= slots_ && n <= capacity_ - position,
        "storage: update must be dense and within capacity/max_write");
    detail::require(
        retains(position, position), "storage: update start has been evicted");
    if (!kv) {
      return;
    }
    detail::require(n > 0, "storage: use nullopt for empty K/V");
    const int end = position + n;
    grow(ring_ ? slots_ : end, s);
    const int offset = ring_ ? position % slots_ : position;
    write(buffers_->k, kv->k, offset, s);
    write(buffers_->v, kv->v, offset, s);
    length_ = std::max(length_, end);
    highwater_ = std::max(highwater_, length_);
  }

  // Slice views in logical order, without joining a wrapped interval.
  detail::Segments read_segments(int begin, int end, StreamOrDevice s) const {
    detail::require(
        retains(begin, end), "storage: requested interval is unavailable");
    if (begin == end) {
      return {};
    }
    const int offset = ring_ ? begin % slots_ : begin;
    const int n = end - begin;
    const int first = std::min(n, slots_ - offset);
    detail::Segments segments{
        KV{detail::tokens(buffers_->k, offset, first, s),
           detail::tokens(buffers_->v, offset, first, s)},
        std::nullopt};
    if (first < n) {
      segments.tail =
          KV{detail::tokens(buffers_->k, 0, n - first, s),
             detail::tokens(buffers_->v, 0, n - first, s)};
    }
    return segments;
  }

  // Reads [begin, end) in logical token order, joining wrapped slices only
  // when necessary. Views and joins remain lazy.
  std::optional<KV> read(int begin, int end, StreamOrDevice s) const {
    auto segments = read_segments(begin, end, s);
    if (!segments.tail) {
      return segments.head;
    }
    return KV{
        ::mlx::core::concatenate({segments.head->k, segments.tail->k}, 2, s),
        ::mlx::core::concatenate({segments.head->v, segments.tail->v}, 2, s)};
  }

 private:
  void grow(int needed, StreamOrDevice s) {
    detail::require(
        needed >= 0 && needed <= slots_, "storage: allocation exceeds storage");
    const int current = buffers_ ? buffers_->k.shape(2) : 0;
    if (needed <= current) {
      return;
    }
    int target = std::max(current, std::max(initial_slots_, 1));
    while (target < needed) {
      // Saturate before adding so doubling cannot overflow.
      target += std::min(target, slots_ - target);
    }
    const ::mlx::core::Shape shape{1, heads_, target - current, head_dim_};
    if (!buffers_) {
      // Separate bases permit independent donation on the initial write.
      buffers_ =
          KV{::mlx::core::zeros(shape, dtype_, s),
             ::mlx::core::zeros(shape, dtype_, s)};
    } else {
      const Tensor pad = ::mlx::core::zeros(shape, dtype_, s);
      buffers_ =
          KV{::mlx::core::concatenate({buffers_->k, pad}, 2, s),
             ::mlx::core::concatenate({buffers_->v, pad}, 2, s)};
    }
  }

  // Writes all input tokens starting at physical slot offset, converting to
  // the storage object's dtype. Any tokens past the end wrap to slot 0.
  // Replaces buffer with the lazy update result; does not execute or
  // synchronize.
  void write(Tensor& buffer, const Tensor& input, int offset, StreamOrDevice s)
      const {
    using namespace ::mlx::core;
    const int n = input.shape(2);
    const int first = std::min(n, slots_ - offset);
    const Tensor stored = detail::cast_if_needed(input, dtype_, s);
    buffer = slice_update(
        buffer,
        detail::tokens(stored, 0, first, s),
        {0, 0, offset, 0},
        {1, heads_, offset + first, head_dim_},
        s);
    const int rest = n - first;
    if (rest > 0) {
      buffer = slice_update(
          buffer,
          detail::tokens(stored, first, rest, s),
          {0, 0, 0, 0},
          {1, heads_, rest, head_dim_},
          s);
    }
  }

  int capacity_, max_write_, slots_;
  bool ring_;
  int initial_slots_, heads_, head_dim_;
  ::mlx::core::Dtype dtype_;
  std::optional<KV> buffers_;
  int length_ = 0, highwater_ = 0;
};

// K/V attention behavior and factory for KVAttentionLayer. Counts describe raw
// Q/K/V before staging; policies own splitting, retention and direct attention.
class KVAttentionPolicy : public AttentionLayerFactory {
 private:
  friend class detail::KVAttentionLayer;

  const bool is_causal;
  // Historical update permission is independent of attention causality.
  const bool allows_history_updates;
  virtual int query_tokens(int kv_tokens) const {
    return kv_tokens;
  }
  virtual int persistent_tokens(int kv_tokens) const {
    return kv_tokens;
  }
  virtual int required_history_begin(int position) const = 0;
  virtual int64_t required_slots(int capacity, int max_write) const = 0;

  // Check that the configured storage can satisfy this policy's retention.
  virtual void validate(const Storage& storage) const {
    detail::require(
        storage.slots() >=
            required_slots(storage.capacity(), storage.max_write()),
        "sequence v2: storage cannot cover policy history/write span");
  }

  // Read-only count/storage checks shared by append admission and updates.
  // A successful result is persistent growth, excluding temporary block tokens.
  virtual std::optional<int>
  admission(const Storage& storage, int position, int kv_tokens) const {
    if (kv_tokens <= 0 || position < 0 || position > storage.length()) {
      return std::nullopt;
    }
    const int queries = query_tokens(kv_tokens);
    const int persistent = persistent_tokens(kv_tokens);
    if (queries <= 0 || queries > kv_tokens ||
        (is_causal && queries != kv_tokens) || persistent < 0 ||
        persistent > kv_tokens || persistent > storage.max_write() ||
        persistent > storage.slots() ||
        int64_t{position} + persistent > storage.capacity() ||
        int64_t{position} + kv_tokens > std::numeric_limits<int>::max() ||
        !storage.retains(required_history_begin(position), position)) {
      return std::nullopt;
    }
    return persistent;
  }

  // Validate without mutation and return the persistent token count.
  // complete_prefix is the cache's last fully constructed multi-layer prefix.
  virtual int validate(
      const Storage& storage,
      int position,
      int complete_prefix,
      const Tensor& q,
      const KV& incoming,
      float scale) const {
    using detail::require;
    if (allows_history_updates) {
      require(
          position >= 0 && position <= storage.length(),
          "sequence v2: aligned update must not leave a hole");
    } else {
      require(
          position >= 0 && position == complete_prefix,
          "sequence v2: update must start at complete prefix");
    }
    require(
        q.ndim() == 4 && q.shape(0) == 1 && q.shape(1) > 0 &&
            q.shape(1) % storage.heads() == 0 && q.shape(2) > 0 &&
            q.shape(3) == storage.head_dim() && detail::floating(q.dtype()) &&
            std::isfinite(scale),
        "sequence v2: invalid query geometry, dtype or scale");
    for (const Tensor* t : {&incoming.k, &incoming.v}) {
      require(
          t->ndim() == 4 && t->shape(0) == 1 &&
              t->shape(1) == storage.heads() &&
              t->shape(3) == storage.head_dim() && t->shape(2) > 0 &&
              t->dtype() == q.dtype(),
          "sequence v2: invalid K/V geometry or dtype");
    }
    const int n = incoming.k.shape(2);
    require(
        incoming.v.shape(2) == n && q.shape(2) == query_tokens(n),
        "sequence v2: K/V or policy query counts differ");
    const auto persistent = admission(storage, position, n);
    require(
        persistent.has_value(),
        "sequence v2: invalid policy counts, capacity/write limits, positions "
        "or retained prefix");
    return *persistent;
  }

  virtual Tensor update_and_attend(
      Storage& storage,
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      StreamOrDevice s,
      std::optional<detail::CachedMask>& mask) const = 0;

 protected:
  explicit KVAttentionPolicy(bool causal, bool history_updates = false)
      : is_causal(causal), allows_history_updates(history_updates) {}

 private:
  std::unique_ptr<AttentionLayer> create_layer(
      const std::shared_ptr<const AttentionLayerFactory>& owner,
      const cache::LayerGeometry& geometry,
      const cache::CacheConfig& cfg) const final;
};

class Full final : public KVAttentionPolicy {
 public:
  Full() : KVAttentionPolicy(true, true) {}

 private:
  int required_history_begin(int /*position*/) const override {
    return 0;
  }
  int64_t required_slots(int capacity, int /*max_write*/) const override {
    return capacity;
  }
  Tensor update_and_attend(
      Storage& storage,
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      StreamOrDevice s,
      std::optional<detail::CachedMask>& /*mask*/) const override {
    storage.update(position, incoming, s);
    const int end = position + incoming.k.shape(2);
    const KV kv = *storage.read(0, end, s);
    return ::mlx::core::fast::scaled_dot_product_attention(
        q,
        detail::cast_if_needed(kv.k, q.dtype(), s),
        detail::cast_if_needed(kv.v, q.dtype(), s),
        scale,
        "causal",
        std::nullopt,
        std::nullopt,
        false,
        s);
  }
};

class SWA final : public KVAttentionPolicy {
 public:
  explicit SWA(int window) : KVAttentionPolicy(true, true), window_(window) {
    detail::require(window > 0, "SWA: window must be positive");
  }

 private:
  int required_history_begin(int position) const override {
    return std::max(0, position - window_ + 1);
  }
  int64_t required_slots(int capacity, int max_write) const override {
    return std::min<int64_t>(capacity, int64_t{window_} + max_write - 1);
  }
  Tensor update_and_attend(
      Storage& storage,
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      StreamOrDevice s,
      std::optional<detail::CachedMask>& cached_mask) const override {
    using namespace ::mlx::core;
    const int begin = required_history_begin(position);
    storage.update(position, incoming, s);
    const int end = position + incoming.k.shape(2);
    const KV kv = *storage.read(begin, end, s);
    const int T = q.shape(2), S = kv.k.shape(2);
    const bool windowed = T > 1 && window_ < S;
    std::optional<Tensor> mask;
    if (windowed) {
      const Stream stream = to_stream(s);
      if (!cached_mask || cached_mask->T != T || cached_mask->S != S ||
          cached_mask->window != window_ || !(cached_mask->stream == stream)) {
        check_allocation_bounded({T, S}, int32, "SWA mask");
        // Lower-right causal alignment, additionally bounded on the left.
        Tensor diff = subtract(
            reshape(arange(S, int32, stream), {1, S}, stream),
            reshape(arange(T, int32, stream), {T, 1}, stream),
            stream);
        Tensor built = reshape(
            logical_and(
                less_equal(diff, array(S - T), stream),
                greater_equal(diff, array(S - T - window_ + 1), stream),
                stream),
            {1, 1, T, S},
            stream);
        cached_mask =
            detail::CachedMask{T, S, window_, std::move(built), stream};
      }
      mask = cached_mask->mask;
    }
    return fast::scaled_dot_product_attention(
        q,
        detail::cast_if_needed(kv.k, q.dtype(), s),
        detail::cast_if_needed(kv.v, q.dtype(), s),
        scale,
        T == 1 || windowed ? "" : "causal",
        mask,
        std::nullopt,
        false,
        s);
  }

  const int window_;
};

// Incoming K/V are [new context, block]. Positive blocks require exactly
// dflash_block_size queries; zero persists all K/V with one dummy query.
// Block size comes from shared cache geometry; retention is layer-specific.
class DFlash final : public KVAttentionPolicy {
 public:
  explicit DFlash(
      const cache::CacheGeometry& geometry,
      std::optional<int> window = std::nullopt)
      : KVAttentionPolicy(false),
        block_size_(geometry.dflash_block_size.value_or(-1)),
        window_(window) {
    detail::require(
        block_size_ >= 0,
        "DFlash: geometry must specify a nonnegative block size");
    detail::require(
        !window_ || *window_ > 0, "DFlash: window must be positive");
  }

 private:
  int query_tokens(int /*kv_tokens*/) const override {
    return block_size_ > 0 ? block_size_ : 1;
  }
  int persistent_tokens(int kv_tokens) const override {
    return kv_tokens - block_size_;
  }
  int required_history_begin(int position) const override {
    return window_ ? std::max(0, position - *window_) : 0;
  }
  int64_t required_slots(int capacity, int /*max_write*/) const override {
    return window_ ? std::min(capacity, *window_) : capacity;
  }
  Tensor update_and_attend(
      Storage& storage,
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      StreamOrDevice s,
      std::optional<detail::CachedMask>& /*mask*/) const override {
    const int context = persistent_tokens(incoming.k.shape(2));
    std::optional<KV> prefix;
    if (context > 0) {
      prefix = block_size_ == 0 ? incoming
                                : KV{detail::tokens(incoming.k, 0, context, s),
                                     detail::tokens(incoming.v, 0, context, s)};
    }
    storage.update(position, prefix, s);
    const std::optional<KV> block = block_size_ > 0
        ? std::optional<KV>(
              KV{detail::tokens(incoming.k, context, block_size_, s),
                 detail::tokens(incoming.v, context, block_size_, s)})
        : std::nullopt;
    const int end = position + context;
    const auto segments =
        storage.read_segments(required_history_begin(end), end, s);
    const KV kv = [&]() -> KV {
      if (!segments.tail && !(segments.head && block)) {
        const auto& part = segments.head ? segments.head : block;
        detail::require(part.has_value(), "DFlash: empty attention interval");
        return KV{
            detail::cast_if_needed(part->k, q.dtype(), s),
            detail::cast_if_needed(part->v, q.dtype(), s)};
      }
      std::vector<Tensor> keys, values;
      keys.reserve(3);
      values.reserve(3);
      // Cast every history segment and the block before the single join.
      for (const auto* part : {&segments.head, &segments.tail, &block}) {
        if (*part) {
          keys.push_back(detail::cast_if_needed((*part)->k, q.dtype(), s));
          values.push_back(detail::cast_if_needed((*part)->v, q.dtype(), s));
        }
      }
      return KV{
          ::mlx::core::concatenate(keys, 2, s),
          ::mlx::core::concatenate(values, 2, s)};
    }();
    return ::mlx::core::fast::scaled_dot_product_attention(
        q, kv.k, kv.v, scale, "", std::nullopt, std::nullopt, false, s);
  }

  const int block_size_;
  const std::optional<int> window_;
};

namespace detail {

class KVAttentionLayer final : public AttentionLayer {
 public:
  KVAttentionLayer(
      std::shared_ptr<const KVAttentionPolicy> policy,
      const cache::LayerGeometry& geometry,
      const cache::CacheConfig& cfg)
      : policy_(std::move(policy)), storage_(geometry, cfg) {
    require(policy_ != nullptr, "sequence v2: null KV policy");
    policy_->validate(storage_);
  }

 private:
  int length() const override {
    return storage_.length();
  }
  QuerySpan query_span(int kv_tokens) const override {
    require(kv_tokens > 0, "sequence v2: query span needs positive K/V count");
    const int count = policy_->query_tokens(kv_tokens);
    require(
        count > 0 && count <= kv_tokens &&
            (!policy_->is_causal || count == kv_tokens),
        "sequence v2: invalid policy query span");
    return {kv_tokens - count, count};
  }
  std::optional<int> admission(int position, int kv_tokens) const override {
    return policy_->admission(storage_, position, kv_tokens);
  }
  bool can_extend(int n) const override {
    return n >= 0 && n <= storage_.capacity() - length() &&
        storage_.retains(policy_->required_history_begin(length()), length());
  }
  bool can_rewind(int position) const override {
    if (position < 0 || position > length()) {
      return false;
    }
    return storage_.can_rewind(
        position,
        position == 0 ? 0 : policy_->required_history_begin(position));
  }
  void rewind(int position) override {
    storage_.rewind(
        position,
        position == 0 ? 0 : policy_->required_history_begin(position));
  }
  void clear() override {
    storage_.clear();
  }
  std::unique_ptr<AttentionLayer> clone_current() const override {
    return std::make_unique<KVAttentionLayer>(*this);
  }
  UpdateInfo validate(
      int position,
      int complete_prefix,
      const Tensor& q,
      const KV& incoming,
      float scale) const override {
    const int persistent = policy_->validate(
        storage_, position, complete_prefix, q, incoming, scale);
    require(
        position >= 0 && persistent >= 0 &&
            int64_t{position} + persistent <= storage_.capacity(),
        "sequence v2: invalid KV update end");
    const int end = position + persistent;
    return {end, policy_->allows_history_updates && end <= length()};
  }
  Tensor update_and_attend(
      int position,
      const Tensor& q,
      const KV& incoming,
      float scale,
      const UpdateInfo& update,
      StreamOrDevice s,
      std::optional<CachedMask>& mask) override {
    Storage staged = storage_;
    Tensor output = policy_->update_and_attend(
        staged, position, q, incoming, scale, s, mask);
    require(
        staged.length() == std::max(length(), update.end),
        "sequence v2: policy changed persistent growth");
    static_assert(std::is_nothrow_move_assignable_v<Storage>);
    storage_ = std::move(staged);
    return output;
  }

  std::shared_ptr<const KVAttentionPolicy> policy_;
  Storage storage_;
};

struct LayerEntry {
  std::unique_ptr<AttentionLayer> attention;
  bool visited = false;
};

} // namespace detail

inline std::unique_ptr<AttentionLayer> KVAttentionPolicy::create_layer(
    const std::shared_ptr<const AttentionLayerFactory>& owner,
    const cache::LayerGeometry& geometry,
    const cache::CacheConfig& cfg) const {
  return std::make_unique<detail::KVAttentionLayer>(
      std::shared_ptr<const KVAttentionPolicy>(owner, this), geometry, cfg);
}

} // namespace sequence_v2

// Single-sequence implementation used by MLXBatchedCache.
// Capacity limits persistent logical length per sequence, not temporary blocks.
// Example (one layer, flat storage, four-token draft block):
//   geometry.dflash_block_size = 4;
//   MLXSequenceCacheV2 c(geometry, cfg);
//   Tensor out = c.attend(0, positions, q, k, v, scale, stream);
// K/V contain [new context, block]; positions cover both, and Q covers the
// block.
//
// A layer ID selects independently owned state. Full/SWA allow retained
// overwrites, writing supplied K/V and preserving later cached values;
// attention is bounded by the call's persistent end, excluding that later
// suffix. Overwrites do not change forward bookkeeping. A forward advances each
// state once, in any order, to an identical persistent end. DFlash keeps one
// visit per forward even with zero growth. Mixed policies require aligned
// logical growth. Updating other layers after a semantic prefix change is the
// caller's duty. length() reports the last fully constructed forward, not GPU
// completion. Graph-construction failures preserve the failing layer, not
// earlier layers. Execution failures cannot roll back published state;
// clear() drops its roots.
//
// Inputs must use MLX-managed storage, not non-owning external-memory wrappers.
// Graphs retain the storage they need. Execution and synchronization belong to
// the caller; neither attend() nor clear() executes work or waits.
// Calls and lifecycle controls must be externally serialized.
class MLXSequenceCacheV2 final : public MLXCache,
                                 public cache::SequenceControl {
 public:
  MLXSequenceCacheV2(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg)
      : MLXSequenceCacheV2(geometry, cfg, make_factories(geometry)) {}

  MLXSequenceCacheV2(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      std::vector<std::shared_ptr<const sequence_v2::AttentionLayerFactory>>
          factories)
      : capacity_(cfg.capacity) {
    using sequence_v2::detail::require;
    require(cache::valid(cfg), "sequence v2: invalid config");
    require(!geometry.layers.empty(), "sequence v2: no layers");
    require(
        factories.size() == geometry.layers.size(),
        "sequence v2: factory count");
    layers_.reserve(geometry.layers.size());
    for (size_t l = 0; l < geometry.layers.size(); ++l) {
      auto attention = sequence_v2::AttentionLayerFactory::create(
          std::move(factories[l]), geometry.layers[l], cfg);
      layers_.push_back({std::move(attention)});
    }
  }

  // Lazy, independently writable fork; a current snapshot needs no rewind.
  MLXSequenceCacheV2(const MLXSequenceCacheV2& source, int keep)
      : capacity_(source.capacity_) {
    using sequence_v2::detail::require;
    require(source.can_fork(keep), "sequence v2: unavailable fork");
    cached_mask_ = source.cached_mask_;
    layers_.reserve(source.layers_.size());
    for (const auto& layer : source.layers_) {
      auto attention = layer.attention->clone_current();
      require(
          attention != nullptr && attention->length() == source.length_,
          "sequence v2: invalid cloned layer");
      if (keep != source.length_) {
        require(
            attention->can_rewind(keep), "sequence v2: clone cannot rewind");
        attention->rewind(keep);
        require(
            attention->length() == keep, "sequence v2: invalid clone rewind");
      }
      layers_.push_back({std::move(attention)});
    }
    reset_forward(keep);
  }

  MLXSequenceCacheV2(const MLXSequenceCacheV2&) = delete;
  MLXSequenceCacheV2& operator=(const MLXSequenceCacheV2&) = delete;

  // Maximum persistent logical length of this sequence.
  int capacity() const override {
    return capacity_;
  }
  int length() const {
    return length_;
  }
  using QuerySpan = sequence_v2::AttentionLayer::QuerySpan;

  // Layout only: valid for placed overwrites and unfinished forwards too.
  QuerySpan query_span(int layer, int kv_tokens) const {
    using sequence_v2::detail::require;
    const auto& attention = layers_[checked_layer(layer)].attention;
    require(kv_tokens > 0, "sequence v2: query span needs positive K/V count");
    const auto span = attention->query_span(kv_tokens);
    require(
        span.begin >= 0 && span.count > 0 &&
            int64_t{span.begin} + span.count <= kv_tokens,
        "sequence v2: invalid layer query span");
    return span;
  }

  // Append preflight only; no tensors are constructed or evaluated.
  // Temporary block tokens do not consume this sequence's persistent capacity.
  bool can_admit(int kv_tokens) const {
    if (kv_tokens <= 0 || pending_end_ ||
        int64_t{length_} + kv_tokens > std::numeric_limits<int>::max()) {
      return false;
    }
    std::optional<int> growth;
    for (const auto& layer : layers_) {
      const auto admitted = layer.attention->admission(length_, kv_tokens);
      if (!admitted || *admitted < 0 || *admitted > kv_tokens ||
          *admitted > capacity_ - length_ || (growth && *growth != *admitted)) {
        return false;
      }
      growth = admitted;
    }
    return true;
  }

  bool can_extend(int n = 1) const override {
    if (pending_end_ || n < 0 || n > capacity_ - length_) {
      return false;
    }
    for (const auto& layer : layers_) {
      if (!layer.attention->can_extend(n)) {
        return false;
      }
    }
    return true;
  }
  bool can_fork(int keep) const {
    return !pending_end_ && keep >= 0 && keep <= length_ &&
        (keep == length_ || can_rewind(keep));
  }
  bool can_rewind(int position) const {
    if (position == 0) {
      return true;
    }
    if (pending_end_ || position < 0 || position > length_) {
      return false;
    }
    for (const auto& layer : layers_) {
      if (!layer.attention->can_rewind(position)) {
        return false;
      }
    }
    return true;
  }
  // Zero resets all states even in an unfinished forward.
  bool rewind(int position) override {
    if (!can_rewind(position)) {
      return false;
    }
    for (auto& layer : layers_) {
      layer.attention->rewind(position);
    }
    reset_forward(position);
    return true;
  }
  void clear() override {
    for (auto& layer : layers_) {
      layer.attention->clear();
    }
    cached_mask_.reset();
    reset_forward(0);
  }

  Tensor attend(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float scale,
      StreamOrDevice s) override {
    using sequence_v2::detail::require;
    const size_t l = checked_layer(layer);
    require(!positions.empty(), "sequence v2: positions must not be empty");
    const sequence_v2::KV incoming{k, v};
    require(
        k.ndim() > 2 && k.shape(2) > 0 &&
            positions.size() == static_cast<size_t>(k.shape(2)),
        "sequence v2: expected one position per incoming K/V token");
    require(
        positions.front() >= 0 &&
            int64_t{positions.front()} + k.shape(2) <=
                std::numeric_limits<int>::max(),
        "sequence v2: incoming K/V positions exceed integer bounds");
    for (size_t i = 0; i < positions.size(); ++i) {
      require(
          int64_t{positions[i]} ==
              int64_t{positions.front()} + static_cast<int64_t>(i),
          "sequence v2: positions must be contiguous and dense");
    }
    const auto update = layers_[l].attention->validate(
        positions.front(), length_, q, incoming, scale);
    return attend_validated(
        l, positions.front(), q, incoming, update, scale, s);
  }

 private:
  static std::vector<std::shared_ptr<const sequence_v2::AttentionLayerFactory>>
  make_factories(const cache::CacheGeometry& geometry) {
    std::vector<std::shared_ptr<const sequence_v2::AttentionLayerFactory>>
        factories;
    factories.reserve(geometry.layers.size());
    for (const auto& layer : geometry.layers) {
      const bool windowed = layer.policy.kind == cache::LayerPolicy::Kind::Ring;
      if (geometry.dflash_block_size.has_value()) {
        factories.push_back(std::make_shared<sequence_v2::DFlash>(
            geometry,
            windowed ? std::optional<int>(layer.policy.window) : std::nullopt));
      } else if (windowed) {
        factories.push_back(
            std::make_shared<sequence_v2::SWA>(layer.policy.window));
      } else {
        factories.push_back(std::make_shared<sequence_v2::Full>());
      }
    }
    return factories;
  }

  // Includes persistent growth in a partially constructed forward.
  int reserved_length() const {
    return pending_end_.value_or(length_);
  }

  void reset_forward(int length) noexcept {
    length_ = length;
    pending_end_.reset();
    visited_count_ = 0;
    for (auto& layer : layers_) {
      layer.visited = false;
    }
  }

  Tensor attend_validated(
      size_t l,
      int position,
      const Tensor& q,
      const sequence_v2::KV& incoming,
      const sequence_v2::AttentionLayer::UpdateInfo& update,
      float scale,
      StreamOrDevice s) {
    using sequence_v2::detail::require;
    auto& layer = layers_[l];
    require(
        update.end >= position && update.end <= capacity_ &&
            int64_t{update.end} - position <= incoming.k.shape(2) &&
            (update.overwrite ? update.end <= layer.attention->length()
                              : update.end >= length_),
        "sequence v2: invalid layer update end");
    if (!update.overwrite) {
      require(!layer.visited, "sequence v2: repeated layer within forward");
      require(
          !pending_end_ || *pending_end_ == update.end,
          "sequence v2: unequal persistent ends");
    }
    Tensor output = layer.attention->update_and_attend(
        position, q, incoming, scale, update, s, cached_mask_);
    if (!update.overwrite) {
      pending_end_ = update.end;
      layer.visited = true;
      if (++visited_count_ == layers_.size()) {
        reset_forward(update.end);
      }
    }
    return output;
  }

  size_t checked_layer(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) {
      throw std::out_of_range("sequence v2: layer out of range");
    }
    return static_cast<size_t>(layer);
  }

  int capacity_;
  std::vector<sequence_v2::detail::LayerEntry> layers_;
  std::optional<sequence_v2::detail::CachedMask> cached_mask_;
  size_t visited_count_ = 0;
  int length_ = 0;
  std::optional<int> pending_end_;
};

} // namespace executorch::backends::mlx
