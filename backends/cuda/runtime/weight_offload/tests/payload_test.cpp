/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Host-level (no CUDA, no GPU) tests for the weight-offload payload
// parser -- the single trust boundary the runtime relies on
// (``CudaBackend::init`` and ``Session::create`` trust the parsed
// struct without re-validating). ``payload.h`` is header-only and
// pulls in nothing CUDA, so these run anywhere.
//
// The wire format mirrors
// ``backends/cuda/passes/weight_offload_pass.py::_serialize_payload``.
// Tests build little-endian bytes directly (the parser uses memcpy
// reads, so this is correct on the LE hosts ExecuTorch builds on) and
// assert each per-field bound and cross-field invariant rejects.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <executorch/backends/cuda/runtime/weight_offload/payload.h>
#include <executorch/runtime/core/error.h>

namespace {

using executorch::runtime::Error;
namespace wo = executorch::backends::cuda::weight_offload;

// c10::ScalarType codes used below (match _TORCH_DTYPE_TO_C10).
constexpr int32_t kInt8 = 1; // element_size 1
constexpr int32_t kFloat32 = 6; // element_size 4

// Little-endian byte accumulator mirroring the Python serializer.
struct Buf {
  std::vector<uint8_t> bytes;

  void raw(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    bytes.insert(bytes.end(), b, b + n);
  }
  void u32(uint32_t v) {
    raw(&v, sizeof(v));
  }
  void u64(uint64_t v) {
    raw(&v, sizeof(v));
  }
  void i32(int32_t v) {
    raw(&v, sizeof(v));
  }
  void i64(int64_t v) {
    raw(&v, sizeof(v));
  }
  // Length-prefixed string with an EXPLICIT length, so tests can lie
  // about the length to exercise the bounds checks.
  void lenstr(uint32_t len, const std::string& s) {
    u32(len);
    raw(s.data(), s.size());
  }
  void str(const std::string& s) {
    lenstr(static_cast<uint32_t>(s.size()), s);
  }
};

// Append a constants_metadata entry. Every field is a parameter so a
// test can emit a single inconsistency (bad strides, wrong nbytes,
// wrong device, ...) on top of an otherwise-valid entry.
void meta(
    Buf& b,
    const std::string& fqn,
    int32_t dtype,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t storage_offset,
    uint64_t nbytes,
    int32_t device_type,
    int32_t device_index) {
  b.str(fqn);
  b.i32(dtype);
  b.u32(static_cast<uint32_t>(sizes.size()));
  for (int64_t s : sizes) {
    b.i64(s);
  }
  for (int64_t s : strides) {
    b.i64(s);
  }
  b.i64(storage_offset);
  b.u64(nbytes);
  b.i32(device_type);
  b.i32(device_index);
}

// Canonical valid payload: method "m", floor 64, schedule [a, b, a],
// no pins, metadata for a (float32 [2,3] = 24B) and b (int8 [4] = 4B).
Buf valid() {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2); // schema_version
  b.str("m");
  b.u64(64); // floor_bytes
  b.u32(3); // schedule_count
  b.str("a");
  b.str("b");
  b.str("a");
  b.u32(0); // pin_count
  b.u32(2); // constants_metadata_count
  meta(b, "a", kFloat32, {2, 3}, {3, 1}, 0, 24, 1, 0);
  meta(b, "b", kInt8, {4}, {1}, 0, 4, 1, 0);
  return b;
}

wo::Payload parse_ok(const Buf& b, const std::string& method = "m") {
  auto r = wo::parse_payload(b.bytes.data(), b.bytes.size(), method);
  EXPECT_TRUE(r.ok()) << "expected parse to succeed";
  return r.ok() ? r.get() : wo::Payload{};
}

void parse_rejects(const Buf& b, const std::string& method = "m") {
  auto r = wo::parse_payload(b.bytes.data(), b.bytes.size(), method);
  EXPECT_FALSE(r.ok()) << "expected parse to reject";
  if (!r.ok()) {
    EXPECT_EQ(r.error(), Error::InvalidArgument);
  }
}

// --------------------------------------------------------------------
// Happy paths
// --------------------------------------------------------------------

TEST(WeightOffloadPayload, ValidPopulatedRoundTrips) {
  wo::Payload p = parse_ok(valid());
  EXPECT_EQ(p.schema_version, 2u);
  EXPECT_EQ(p.method_name, "m");
  EXPECT_EQ(p.floor_bytes, 64u);
  ASSERT_EQ(p.schedule.size(), 3u);
  EXPECT_EQ(p.schedule[0], "a");
  EXPECT_EQ(p.schedule[1], "b");
  EXPECT_EQ(p.schedule[2], "a");
  EXPECT_TRUE(p.pin_fqns.empty());
  ASSERT_EQ(p.constants_metadata.size(), 2u);
  // first-seen order preserved
  EXPECT_EQ(p.constants_metadata[0].fqn, "a");
  EXPECT_EQ(p.constants_metadata[0].nbytes, 24u);
  EXPECT_EQ(p.constants_metadata[1].fqn, "b");
  EXPECT_EQ(p.constants_metadata[1].nbytes, 4u);
}

TEST(WeightOffloadPayload, EmptyScheduleIsValid) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(0); // schedule
  b.u32(0); // pin
  b.u32(0); // metadata
  wo::Payload p = parse_ok(b);
  EXPECT_TRUE(p.schedule.empty());
  EXPECT_TRUE(p.constants_metadata.empty());
}

TEST(WeightOffloadPayload, ScalarConstantAccepted) {
  // ndim 0: no sizes/strides, logical == element_size, numel 1.
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("s");
  b.u32(0); // pin
  b.u32(1); // metadata
  meta(b, "s", kFloat32, {}, {}, 0, 4, 1, 0);
  wo::Payload p = parse_ok(b);
  ASSERT_EQ(p.constants_metadata.size(), 1u);
  EXPECT_EQ(p.constants_metadata[0].nbytes, 4u);
}

TEST(WeightOffloadPayload, PinSubsetOfScheduleAccepted) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(24);
  b.u32(2);
  b.str("a");
  b.str("b");
  b.u32(1); // pin
  b.str("a");
  b.u32(2);
  meta(b, "a", kFloat32, {2, 3}, {3, 1}, 0, 24, 1, 0);
  meta(b, "b", kInt8, {4}, {1}, 0, 4, 1, 0);
  wo::Payload p = parse_ok(b);
  ASSERT_EQ(p.pin_fqns.size(), 1u);
  EXPECT_EQ(p.pin_fqns[0], "a");
}

TEST(WeightOffloadPayload, EmptyExpectedMethodAcceptsAny) {
  // expected_method "" disables the method-name match.
  auto b = valid();
  auto r = wo::parse_payload(b.bytes.data(), b.bytes.size(), "");
  EXPECT_TRUE(r.ok());
}

// --------------------------------------------------------------------
// Framing / header
// --------------------------------------------------------------------

TEST(WeightOffloadPayload, RejectsBadMagic) {
  auto b = valid();
  // Corrupt the first 4 bytes.
  uint32_t bad = 0xDEADBEEF;
  std::memcpy(b.bytes.data(), &bad, sizeof(bad));
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsWrongSchemaVersion) {
  for (uint32_t ver : {1u, 3u, 0u}) {
    Buf b = valid();
    std::memcpy(b.bytes.data() + 4, &ver, sizeof(ver));
    parse_rejects(b);
  }
}

TEST(WeightOffloadPayload, RejectsTruncationAtEveryPrefix) {
  auto full = valid();
  // Every strict prefix of a valid payload must be rejected (no
  // truncation read past the end). Skip length 0 trivially-empty.
  for (size_t n = 1; n < full.bytes.size(); ++n) {
    auto r = wo::parse_payload(full.bytes.data(), n, "m");
    EXPECT_FALSE(r.ok()) << "prefix length " << n << " should reject";
  }
}

TEST(WeightOffloadPayload, RejectsTrailingBytes) {
  auto b = valid();
  b.bytes.push_back(0x00); // one extra byte
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsMethodNameMismatch) {
  parse_rejects(valid(), "not_m");
}

// --------------------------------------------------------------------
// Bounds
// --------------------------------------------------------------------

TEST(WeightOffloadPayload, RejectsOverlongString) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  // method_name length claims > kMaxStrLen.
  b.lenstr(wo::kMaxStrLen + 1, std::string(wo::kMaxStrLen + 1, 'x'));
  b.u64(0);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsScheduleCountOverCap) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(wo::kMaxScheduleCount + 1); // rejected before any entry is read
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsPinCountOverCap) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(0); // schedule
  b.u32(wo::kMaxPinCount + 1);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsMetadataCountOverCap) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(0); // schedule
  b.u32(0); // pin
  b.u32(wo::kMaxConstantsMetadataCount + 1);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsNdimOverCap) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("a");
  b.u32(0); // pin
  b.u32(1); // metadata
  b.str("a");
  b.i32(kFloat32);
  b.u32(wo::kMaxNdim + 1); // ndim over cap, rejected before reading sizes
  parse_rejects(b);
}

// --------------------------------------------------------------------
// Per-FQN metadata invariants
// --------------------------------------------------------------------

// Build a single-constant payload whose lone metadata entry is emitted
// by ``emit`` (so a test can inject exactly one inconsistency).
template <typename EmitFn>
Buf one_const(EmitFn emit) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1); // schedule
  b.str("a");
  b.u32(0); // pin
  b.u32(1); // metadata
  emit(b);
  return b;
}

TEST(WeightOffloadPayload, RejectsEmptyFqnInMetadata) {
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "", kFloat32, {2}, {1}, 0, 8, 1, 0); }));
}

TEST(WeightOffloadPayload, RejectsUnsupportedDtype) {
  // 7, 8, 9, 10 are not in the allow-list (element_size returns 0).
  for (int32_t dt : {7, 8, 9, 10, 99, -1}) {
    parse_rejects(one_const(
        [dt](Buf& b) { meta(b, "a", dt, {2}, {1}, 0, 8, 1, 0); }));
  }
}

TEST(WeightOffloadPayload, RejectsNonPositiveSize) {
  parse_rejects(
      one_const([](Buf& b) { meta(b, "a", kFloat32, {0}, {1}, 0, 0, 1, 0); }));
  parse_rejects(
      one_const([](Buf& b) { meta(b, "a", kFloat32, {-2}, {1}, 0, 8, 1, 0); }));
}

TEST(WeightOffloadPayload, RejectsNonContiguousStrides) {
  // [2,3] contiguous strides are {3,1}; {1,2} is not C-contiguous.
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2, 3}, {1, 2}, 0, 24, 1, 0); }));
}

TEST(WeightOffloadPayload, RejectsNonZeroStorageOffset) {
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2, 3}, {3, 1}, 1, 24, 1, 0); }));
}

TEST(WeightOffloadPayload, RejectsNbytesMismatch) {
  // float32 [2,3] logical == 24; claim 23 and 25.
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2, 3}, {3, 1}, 0, 23, 1, 0); }));
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2, 3}, {3, 1}, 0, 25, 1, 0); }));
}

TEST(WeightOffloadPayload, RejectsWrongDeviceType) {
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2}, {1}, 0, 8, 0, 0); })); // cpu
}

TEST(WeightOffloadPayload, RejectsWrongDeviceIndex) {
  parse_rejects(one_const(
      [](Buf& b) { meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 1); })); // cuda:1
}

// --------------------------------------------------------------------
// Cross-field invariants
// --------------------------------------------------------------------

TEST(WeightOffloadPayload, RejectsScheduleMetadataSetMismatch) {
  // schedule {a} but metadata describes {b}.
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("a");
  b.u32(0);
  b.u32(1);
  meta(b, "b", kFloat32, {2}, {1}, 0, 8, 1, 0);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsMetadataMissingScheduledFqn) {
  // schedule {a, b} but metadata only {a}.
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(2);
  b.str("a");
  b.str("b");
  b.u32(0);
  b.u32(1);
  meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 0);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsDuplicateFqnInMetadata) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("a");
  b.u32(0);
  b.u32(2); // two metadata entries, both "a"
  meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 0);
  meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 0);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsDuplicatePinFqn) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("a");
  b.u32(2); // pin "a" twice
  b.str("a");
  b.str("a");
  b.u32(1);
  meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 0);
  parse_rejects(b);
}

TEST(WeightOffloadPayload, RejectsPinFqnNotInSchedule) {
  Buf b;
  b.u32(wo::kPayloadMagic);
  b.u32(2);
  b.str("m");
  b.u64(0);
  b.u32(1);
  b.str("a");
  b.u32(1); // pin "z" which is not scheduled
  b.str("z");
  b.u32(1);
  meta(b, "a", kFloat32, {2}, {1}, 0, 8, 1, 0);
  parse_rejects(b);
}

// --------------------------------------------------------------------
// named_data_key_for_method must match the Python helper format.
// --------------------------------------------------------------------

TEST(WeightOffloadPayload, NamedDataKeyFormat) {
  EXPECT_EQ(wo::named_data_key_for_method("forward"),
            "_weight_offload_payload__forward");
  EXPECT_EQ(wo::named_data_key_for_method("decode"),
            "_weight_offload_payload__decode");
}

} // namespace
