/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ===========================================================================
// EXPERIMENTAL -- PRIVATE WIRE FORMAT, V2
// ===========================================================================
// Runtime decoder for the weight-offload payload that
// ``backends/cuda/passes/weight_offload_pass.py::_serialize_payload``
// produces and ``CudaBackend::init`` retrieves from the NamedDataMap
// when the private ``_weight_offload_internal_enable`` compile spec
// is set.
//
// The wire format is a fixed-shape little-endian binary layout (see
// the Python serializer for the authoritative description). JSON
// was rejected to keep the runtime free of parser dependencies for a
// private, fixed-shape payload.
//
// The per-FQN ``constants_metadata`` block carries dtype / sizes /
// strides / device for each scheduled FQN. With AOTI's eager constant
// load skipped, the runtime can no longer recover that metadata from
// extract_constants_map (it would return placeholder metadata for the
// installed dummies), so it has to come over the wire. Schema v1 is
// rejected at parse — maintaining two runtime paths for an
// experimental feature is dead weight.
// ===========================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::cuda::weight_offload {

// Must match the Python ``PAYLOAD_MAGIC`` constant. Spells "ETWO" in
// little-endian ASCII.
inline constexpr uint32_t kPayloadMagic = 0x4F575445;

// Bounded sizes that mirror the Python serializer's limits. Anything
// past these caps is a hard fail — bounded means schema drift or an
// attacker-controlled artifact, neither of which we silently accept.
inline constexpr uint32_t kMaxStrLen = 1024;
inline constexpr uint32_t kMaxScheduleCount = 1'000'000;
inline constexpr uint32_t kMaxPinCount = 100'000;
inline constexpr uint32_t kMaxConstantsMetadataCount = 1'000'000;
inline constexpr uint32_t kMaxNdim = 32;

// Single-device constraint. ``create_with_device(..., "cuda", nullptr)``
// doesn't take a per-method device index; until that's plumbed through,
// dummies + stream + pool land on whichever device happens to be
// current (effectively 0). Asserting at parse surfaces any payload that
// disagrees, instead of silently using device 0.
inline constexpr int32_t kCudaDeviceType = 1;
inline constexpr int32_t kRequiredDeviceIndex = 0;

struct ConstantMetadata {
  std::string fqn;
  int32_t dtype{0};
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset{0};
  uint64_t nbytes{0};
  int32_t device_type{0};
  int32_t device_index{0};
};

struct Payload {
  uint32_t schema_version{0};
  std::string method_name;
  uint64_t floor_bytes{0};
  std::vector<std::string> schedule;
  std::vector<std::string> pin_fqns;
  // V2 block. Indexed by first-seen position in ``schedule``; the
  // parser enforces ``{ entry.fqn } == set(schedule)`` so the runtime
  // can look up by FQN safely.
  std::vector<ConstantMetadata> constants_metadata;
};

namespace detail {

// Tiny cursor over a borrowed byte range. Every read advances the
// offset and bounds-checks against the end; truncation is a single
// error code (``InvalidArgument``).
class Cursor {
 public:
  Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  size_t remaining() const {
    return size_ - offset_;
  }
  size_t offset() const {
    return offset_;
  }
  bool exhausted() const {
    return offset_ == size_;
  }

  ::executorch::runtime::Error read_u32(uint32_t& out) {
    if (remaining() < sizeof(uint32_t)) {
      return ::executorch::runtime::Error::InvalidArgument;
    }
    std::memcpy(&out, data_ + offset_, sizeof(uint32_t));
    offset_ += sizeof(uint32_t);
    return ::executorch::runtime::Error::Ok;
  }

  ::executorch::runtime::Error read_u64(uint64_t& out) {
    if (remaining() < sizeof(uint64_t)) {
      return ::executorch::runtime::Error::InvalidArgument;
    }
    std::memcpy(&out, data_ + offset_, sizeof(uint64_t));
    offset_ += sizeof(uint64_t);
    return ::executorch::runtime::Error::Ok;
  }

  ::executorch::runtime::Error read_i32(int32_t& out) {
    uint32_t tmp = 0;
    auto err = read_u32(tmp);
    if (err != ::executorch::runtime::Error::Ok) {
      return err;
    }
    std::memcpy(&out, &tmp, sizeof(int32_t));
    return ::executorch::runtime::Error::Ok;
  }

  ::executorch::runtime::Error read_i64(int64_t& out) {
    uint64_t tmp = 0;
    auto err = read_u64(tmp);
    if (err != ::executorch::runtime::Error::Ok) {
      return err;
    }
    std::memcpy(&out, &tmp, sizeof(int64_t));
    return ::executorch::runtime::Error::Ok;
  }

  ::executorch::runtime::Error read_bounded_string(
      std::string& out,
      uint32_t max_len) {
    uint32_t len = 0;
    auto err = read_u32(len);
    if (err != ::executorch::runtime::Error::Ok) {
      return err;
    }
    if (len > max_len || len > remaining()) {
      return ::executorch::runtime::Error::InvalidArgument;
    }
    out.assign(reinterpret_cast<const char*>(data_ + offset_), len);
    offset_ += len;
    return ::executorch::runtime::Error::Ok;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t offset_{0};
};

// Element size by dtype code. Returns 0 for unsupported codes, which
// the caller treats as an invalid-payload signal. The supported set
// mirrors the pass-side ``_TORCH_DTYPE_TO_C10`` map; extending one
// without the other will hard-fail at parse, which is the intended
// drift signal.
inline uint64_t element_size(int32_t dtype) {
  switch (dtype) {
    case 0: // uint8
    case 1: // int8
    case 11: // bool
      return 1;
    case 2: // int16
    case 5: // float16
    case 15: // bfloat16
      return 2;
    case 3: // int32
    case 6: // float32
      return 4;
    case 4: // int64
      return 8;
    default:
      return 0;
  }
}

// Read a single ConstantMetadata entry with per-field bounds + cross-
// field consistency (dtype is supported, sizes positive, strides
// describe a C-contiguous layout, storage_offset == 0, nbytes ==
// elementSize(dtype) * product(sizes)). Catching these at parse means
// downstream code can trust the parsed struct directly.
inline ::executorch::runtime::Error read_constant_metadata(
    Cursor& cur,
    ConstantMetadata& m) {
  using ::executorch::runtime::Error;
  if (cur.read_bounded_string(m.fqn, kMaxStrLen) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (m.fqn.empty()) {
    return Error::InvalidArgument;
  }
  if (cur.read_i32(m.dtype) != Error::Ok) {
    return Error::InvalidArgument;
  }
  const uint64_t esize = element_size(m.dtype);
  if (esize == 0) {
    return Error::InvalidArgument;
  }
  uint32_t ndim = 0;
  if (cur.read_u32(ndim) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (ndim > kMaxNdim) {
    return Error::InvalidArgument;
  }
  m.sizes.resize(ndim);
  uint64_t logical = esize;
  for (uint32_t k = 0; k < ndim; ++k) {
    if (cur.read_i64(m.sizes[k]) != Error::Ok) {
      return Error::InvalidArgument;
    }
    // Positive sizes only. Scalars (ndim==0) skip this loop entirely
    // and are accepted: logical stays at element_size, numel == 1.
    if (m.sizes[k] <= 0) {
      return Error::InvalidArgument;
    }
    const uint64_t s_u = static_cast<uint64_t>(m.sizes[k]);
    if (logical > std::numeric_limits<uint64_t>::max() / s_u) {
      return Error::InvalidArgument;
    }
    logical *= s_u;
  }
  m.strides.resize(ndim);
  for (uint32_t k = 0; k < ndim; ++k) {
    if (cur.read_i64(m.strides[k]) != Error::Ok) {
      return Error::InvalidArgument;
    }
  }
  // Strides must describe a C-contiguous layout: strides[i] ==
  // product(sizes[i+1..]). The offload host mirror is sized for
  // logical bytes and the H2D copy is dense, so any non-contiguous
  // layout would over- or under-read.
  {
    int64_t expected = 1;
    for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
      if (m.strides[i] != expected) {
        return Error::InvalidArgument;
      }
      expected *= m.sizes[i];
    }
  }
  if (cur.read_i64(m.storage_offset) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (m.storage_offset != 0) {
    return Error::InvalidArgument;
  }
  if (cur.read_u64(m.nbytes) != Error::Ok) {
    return Error::InvalidArgument;
  }
  // nbytes must equal the logical byte count derived from dtype +
  // sizes. The pass writes it as `element_size * product(sizes)`;
  // catching drift here means downstream consumers can read either
  // field interchangeably.
  if (m.nbytes != logical) {
    return Error::InvalidArgument;
  }
  if (cur.read_i32(m.device_type) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (m.device_type != kCudaDeviceType) {
    return Error::InvalidArgument;
  }
  if (cur.read_i32(m.device_index) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (m.device_index != kRequiredDeviceIndex) {
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

} // namespace detail

// Decode the binary payload at ``[data, data + size)``. Validates
// magic / schema_version / per-field bounds and rejects truncation,
// trailing bytes, and (when non-empty) ``expected_method`` mismatch.
// ``expected_method`` is the method this ``init`` call is for; the
// payload's embedded ``method_name`` must match it.
//
// REJECTS v1 with Error::InvalidArgument so the caller can surface
// a "rebuild required" message. V2-only parser.
inline ::executorch::runtime::Result<Payload> parse_payload(
    const void* data,
    size_t size,
    const std::string& expected_method) {
  using ::executorch::runtime::Error;
  detail::Cursor cur(static_cast<const uint8_t*>(data), size);

  uint32_t magic = 0;
  if (cur.read_u32(magic) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (magic != kPayloadMagic) {
    return Error::InvalidArgument;
  }

  Payload p;
  if (cur.read_u32(p.schema_version) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (p.schema_version != 2) {
    return Error::InvalidArgument;
  }

  if (cur.read_bounded_string(p.method_name, kMaxStrLen) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (!expected_method.empty() && p.method_name != expected_method) {
    return Error::InvalidArgument;
  }

  if (cur.read_u64(p.floor_bytes) != Error::Ok) {
    return Error::InvalidArgument;
  }

  uint32_t schedule_count = 0;
  if (cur.read_u32(schedule_count) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (schedule_count > kMaxScheduleCount) {
    return Error::InvalidArgument;
  }
  p.schedule.reserve(schedule_count);
  for (uint32_t i = 0; i < schedule_count; ++i) {
    std::string s;
    if (cur.read_bounded_string(s, kMaxStrLen) != Error::Ok) {
      return Error::InvalidArgument;
    }
    p.schedule.push_back(std::move(s));
  }

  uint32_t pin_count = 0;
  if (cur.read_u32(pin_count) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (pin_count > kMaxPinCount) {
    return Error::InvalidArgument;
  }
  p.pin_fqns.reserve(pin_count);
  for (uint32_t i = 0; i < pin_count; ++i) {
    std::string s;
    if (cur.read_bounded_string(s, kMaxStrLen) != Error::Ok) {
      return Error::InvalidArgument;
    }
    p.pin_fqns.push_back(std::move(s));
  }

  uint32_t md_count = 0;
  if (cur.read_u32(md_count) != Error::Ok) {
    return Error::InvalidArgument;
  }
  if (md_count > kMaxConstantsMetadataCount) {
    return Error::InvalidArgument;
  }
  p.constants_metadata.resize(md_count);
  for (uint32_t i = 0; i < md_count; ++i) {
    if (detail::read_constant_metadata(cur, p.constants_metadata[i]) !=
        Error::Ok) {
      return Error::InvalidArgument;
    }
  }

  // Trailing bytes are a schema-drift signal — refuse to accept a
  // payload whose serializer wrote extra fields this build doesn't
  // know about.
  if (!cur.exhausted()) {
    return Error::InvalidArgument;
  }

  // Cross-field invariants for v2:
  //   - constants_metadata FQN set must equal unique(schedule).
  //   - No duplicate FQNs across metadata entries.
  //   - No duplicate FQNs in pin_fqns.
  //   - Every pin_fqn must appear in the schedule.
  // Catching these at parse means downstream code (init, Session)
  // can trust the parsed struct without re-validating.
  if (!p.constants_metadata.empty() || !p.schedule.empty()) {
    std::vector<std::string> md_fqns;
    md_fqns.reserve(p.constants_metadata.size());
    for (const auto& m : p.constants_metadata) {
      md_fqns.push_back(m.fqn);
    }
    std::sort(md_fqns.begin(), md_fqns.end());
    for (size_t i = 1; i < md_fqns.size(); ++i) {
      if (md_fqns[i] == md_fqns[i - 1]) {
        return Error::InvalidArgument; // duplicate FQN in metadata
      }
    }
    std::vector<std::string> sched_unique = p.schedule;
    std::sort(sched_unique.begin(), sched_unique.end());
    sched_unique.erase(
        std::unique(sched_unique.begin(), sched_unique.end()),
        sched_unique.end());
    if (md_fqns != sched_unique) {
      return Error::InvalidArgument;
    }
  }
  if (!p.pin_fqns.empty()) {
    std::unordered_set<std::string> sched_set(
        p.schedule.begin(), p.schedule.end());
    std::unordered_set<std::string> pin_set;
    pin_set.reserve(p.pin_fqns.size());
    for (const auto& f : p.pin_fqns) {
      if (!pin_set.insert(f).second) {
        return Error::InvalidArgument; // duplicate in pin_fqns
      }
      if (sched_set.find(f) == sched_set.end()) {
        return Error::InvalidArgument; // pin_fqn not in schedule
      }
    }
  }

  return p;
}

// Same compile-spec key the Python side uses
// (``COMPILE_SPEC_KEY_ENABLE``). Defined here so the runtime checks
// against a single string constant, not a string literal sprinkled
// through ``cuda_backend.cpp``.
inline constexpr const char* kEnableCompileSpecKey =
    "_weight_offload_internal_enable";

// NamedDataMap key prefix; the per-method key is
// ``f"{prefix}__{method_name}"``. Must match the Python helper
// ``named_data_key_for_method``.
inline constexpr const char* kNamedDataKeyPrefix = "_weight_offload_payload";

inline std::string named_data_key_for_method(const std::string& method) {
  std::string key = kNamedDataKeyPrefix;
  key += "__";
  key += method;
  return key;
}

} // namespace executorch::backends::cuda::weight_offload
