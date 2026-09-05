/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023, 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Warning: Do not change this without changing arm_vela.py::vela_compile
 *          as that function emits this format and the two need to align.
 */

#include <executorch/backends/arm/runtime/VelaBinStream.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/named_data_map.h>

namespace executorch {
namespace backends {
namespace arm {

namespace {

using ::executorch::runtime::Error;
using ::executorch::runtime::NamedDataMap;

constexpr size_t kVelaExternalBlockKeySize = 64;

constexpr uintptr_t next_mul_16(uintptr_t value) {
  return ((value - 1) | 15) + 1;
}

struct VelaBlockPayload {
  const char* data;
  size_t size;
};

// Reads the key from block.data and its referenced payload from named_data_map.
Error resolve_external_block(
    const VelaBinBlock& block,
    const NamedDataMap& named_data_map,
    VelaBlockPayload& payload) {
  const std::string_view key(block.data, block.size);
  auto result = named_data_map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  payload.data = static_cast<const char*>(result->data());
  payload.size = result->size();
  if (payload.data == nullptr ||
      reinterpret_cast<uintptr_t>(payload.data) !=
          next_mul_16(reinterpret_cast<uintptr_t>(payload.data))) {
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

// Reinterpret a block payload as a VelaIOs descriptor, but only after checking
// that the payload is large enough to hold the `count` field and the full
// `io[count]` array. Without this a malformed stream could point `count` past
// the buffer, and every subsequent `io[i]` access (including the scratch
// offsets used for memcpy at execute time) would read out of bounds.
Error bind_io_block(const VelaBlockPayload& payload, VelaIOs** out) {
  if (payload.size < sizeof(VelaIOs)) {
    ET_LOG(Error, "Vela IO block too small for count field");
    return Error::InvalidProgram;
  }
  VelaIOs* ios = reinterpret_cast<VelaIOs*>(const_cast<char*>(payload.data));
  if (ios->count < 0) {
    ET_LOG(Error, "Negative Vela IO count %d", ios->count);
    return Error::InvalidProgram;
  }
  const size_t io_bytes = payload.size - sizeof(VelaIOs);
  if (static_cast<size_t>(ios->count) > io_bytes / sizeof(VelaIO)) {
    ET_LOG(
        Error,
        "Vela IO count %d overruns %zu payload bytes",
        ios->count,
        payload.size);
    return Error::InvalidProgram;
  }
  *out = ios;
  return Error::Ok;
}

} // namespace

bool vela_bin_validate(const char* data, int size) {
  // A single block header must fit before computing the footer pointer;
  // otherwise foot = data + size - sizeof(VelaBinBlock) underflows below data
  // and the strncmp below reads out of bounds.
  if (size < static_cast<int>(sizeof(VelaBinBlock))) {
    ET_LOG(Error, "Vela binary too small: %d bytes", size);
    return false;
  }
  const char* foot = data + size - sizeof(VelaBinBlock);

  // Check 16 byte alignment
  bool valid = true;
  if ((uintptr_t)data != next_mul_16((uintptr_t)data)) {
    ET_LOG(Error, "Vela bin ptr not aligned to 16 bytes: %p", data);
    valid = false;
  }
  if ((uintptr_t)foot != next_mul_16((uintptr_t)foot)) {
    ET_LOG(Error, "End of vela bin not aligned to 16 bytes: %p", foot);
    valid = false;
  }
  // Check header and footer blocks are the right format
  if (strncmp(data, "vela_bin_stream", strlen("vela_bin_stream")) != 0) {
    ET_LOG(Error, "Incorrect header in vela_bin_stream");
    valid = false;
  }
  if (strncmp(foot, "vela_end_stream", strlen("vela_end_stream")) != 0) {
    ET_LOG(Error, "Incorrect footer in vela_bin_stream");
    valid = false;
  }

  return valid;
}

Error vela_bin_read(
    const char* data,
    int size,
    const NamedDataMap* named_data_map,
    VelaHandles* handles) {
  const char* ptr = data;
  const size_t total = size < 0 ? 0 : static_cast<size_t>(size);

  while (static_cast<size_t>(ptr - data) < total) {
    // A full block header must be present before any of its fields (size,
    // name, external) are read. The stream is attacker-controllable when the
    // delegate blob comes from a malformed .pte, so never dereference past the
    // buffer.
    const size_t remaining = total - static_cast<size_t>(ptr - data);
    if (remaining < sizeof(VelaBinBlock)) {
      ET_LOG(Error, "Truncated vela block header");
      return Error::InvalidProgram;
    }
    VelaBinBlock* b = reinterpret_cast<VelaBinBlock*>(const_cast<char*>(ptr));
    // The block's inline payload (padded to 16 bytes) must fit within the
    // bytes remaining after its header; otherwise b->data + b->size, and the
    // ptr advance below, would run off the end of the buffer.
    const size_t avail = remaining - sizeof(VelaBinBlock);
    // Check the raw size against the buffer first, in size_t, before padding.
    // next_mul_16() is computed in uintptr_t and wraps to 0 for b->size near
    // UINT32_MAX on 32-bit targets (i.e. Cortex-M), which would otherwise let a
    // ~4GB payload slip past a padded-only check.
    if (b->size > avail) {
      ET_LOG(
          Error,
          "Vela block size %u overruns %zu remaining payload bytes",
          static_cast<unsigned>(b->size),
          avail);
      return Error::InvalidProgram;
    }
    const size_t padded =
        (static_cast<size_t>(b->size) + 15) & ~static_cast<size_t>(15);
    if (padded > avail) {
      ET_LOG(
          Error,
          "Padded vela block size %zu overruns %zu remaining payload bytes",
          padded,
          avail);
      return Error::InvalidProgram;
    }
    ptr += sizeof(VelaBinBlock) + padded;

    VelaBlockPayload payload{b->data, b->size};
    if (b->external == kVelaExternalBlockReference) {
      if (b->size != kVelaExternalBlockKeySize || named_data_map == nullptr) {
        return Error::InvalidProgram;
      }
      const Error status = resolve_external_block(*b, *named_data_map, payload);
      if (status != Error::Ok) {
        return status;
      }
    }
    if (!strncmp(b->name, "vela_bin_stream", strlen("vela_bin_stream"))) {
      // expect vela_bin_stream first
      if (reinterpret_cast<char*>(b) !=
          reinterpret_cast<char*>(const_cast<char*>(data)))
        return Error::InvalidProgram;
    } else if (!strncmp(b->name, "cmd_data", strlen("cmd_data"))) {
      // This driver magic header confirms a valid command stream in binary
      if (payload.size < strlen("COP1")) {
        return Error::InvalidProgram;
      }
      if (strncmp(payload.data, "COP1", strlen("COP1")) &&
          strncmp(payload.data, "COP2", strlen("COP2"))) {
        return Error::InvalidProgram;
      }
      handles->cmd_data = payload.data;
      handles->cmd_data_size = payload.size;
    } else if (!strncmp(b->name, "weight_data", strlen("weight_data"))) {
      handles->weight_data = payload.data;
      handles->weight_data_size = payload.size;
    } else if (!strncmp(b->name, "scratch_size", strlen("scratch_size"))) {
      if (payload.size < sizeof(uint32_t)) {
        ET_LOG(Error, "Malformed scratch_size block");
        return Error::InvalidProgram;
      }
      const uint32_t* scratch_size_ptr =
          reinterpret_cast<const uint32_t*>(payload.data);
      handles->scratch_data_size = *scratch_size_ptr;
    } else if (!strncmp(b->name, "inputs", strlen("inputs"))) {
      const Error status = bind_io_block(payload, &handles->inputs);
      if (status != Error::Ok) {
        return status;
      }
    } else if (!strncmp(b->name, "outputs", strlen("outputs"))) {
      const Error status = bind_io_block(payload, &handles->outputs);
      if (status != Error::Ok) {
        return status;
      }
    } else if (!strncmp(
                   b->name, "vela_end_stream", strlen("vela_end_stream"))) {
      // expect vela_end_stream last
      if (ptr - data != size) {
        ET_LOG(Error, "Expected vela binary to end with vela_end_stream");
        return Error::InvalidProgram;
      }
      return Error::Ok;
    } else {
      // Unrecognised block name
      ET_LOG(Error, "Invalid block name or malformed binary");
      return Error::InvalidProgram;
    }
  }

  // We've fallen off the end without finding vela_end_stream
  return Error::InvalidProgram;
}

bool vela_bin_read(const char* data, VelaHandles* handles, int size) {
  return vela_bin_read(data, size, nullptr, handles) == Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
