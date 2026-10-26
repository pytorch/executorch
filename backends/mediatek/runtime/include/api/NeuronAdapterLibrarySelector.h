/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string_view>

#include <executorch/runtime/backend/interface.h>

namespace executorch {
namespace backends {
namespace neuron {

inline constexpr std::string_view kNeuronAdapterMajorKey =
    "neuron-adapter-major";

struct AdapterLibrarySelection {
  void* handle;
  int major;
};

struct AdapterLibraryState {
  void* handle = nullptr;
  int major = -1;
  bool loadAttempted = false;
};

inline int parseAdapterLibraryMajor(const void* data, std::size_t size) {
  if (data == nullptr || size == 0 || size > 3) {
    return -1;
  }
  int major = 0;
  for (std::size_t index = 0; index < size; ++index) {
    const char digit = static_cast<const char*>(data)[index];
    if (digit < '0' || digit > '9') {
      return -1;
    }
    major = major * 10 + digit - '0';
  }
  return major > 0 && major <= 255 ? major : -1;
}

inline int scanAdapterLibraryMajor(
    runtime::ArrayRef<runtime::CompileSpec> compileSpecs) {
  int requestedMajor = 0;
  bool found = false;
  for (const auto& compileSpec : compileSpecs) {
    if (compileSpec.key == nullptr ||
        std::strcmp(compileSpec.key, kNeuronAdapterMajorKey.data()) != 0) {
      continue;
    }
    if (found) {
      return -1;
    }
    requestedMajor = parseAdapterLibraryMajor(
        compileSpec.value.buffer, compileSpec.value.nbytes);
    if (requestedMajor < 0) {
      return -1;
    }
    found = true;
  }
  return requestedMajor;
}

inline bool isAdapterLibraryMajorCompatible(
    int selectedMajor,
    int requestedMajor) {
  return requestedMajor == 0 || selectedMajor == requestedMajor;
}

template <typename OpenLibrary, typename CloseLibrary, typename GetMajor>
AdapterLibrarySelection selectAdapterLibrary(
    int requestedMajor,
    OpenLibrary openLibrary,
    CloseLibrary closeLibrary,
    GetMajor getMajor) {
  if (requestedMajor < 0 || requestedMajor > 255) {
    return {nullptr, -1};
  }

  std::array<char, 64> versionedName{};
  std::array<const char*, 4> candidates{};
  std::size_t candidateCount = 0;
  if (requestedMajor > 0) {
    std::snprintf(
        versionedName.data(),
        versionedName.size(),
        "libneuronusdk_adapter.%d.mtk.so",
        requestedMajor);
    candidates[candidateCount++] = versionedName.data();
  }
  candidates[candidateCount++] = "libneuronusdk_adapter.mtk.so";
  candidates[candidateCount++] = "libneuron_adapter_mgvi.so";
  candidates[candidateCount++] = "libneuron_adapter.so";

  for (std::size_t index = 0; index < candidateCount; ++index) {
    void* handle = openLibrary(candidates[index]);
    if (handle == nullptr) {
      continue;
    }
    const int runtimeMajor = getMajor(handle);
    if (requestedMajor == 0 || runtimeMajor == requestedMajor) {
      return {handle, runtimeMajor > 0 ? runtimeMajor : 0};
    }
    closeLibrary(handle);
  }
  return {nullptr, -1};
}

template <typename SelectLibrary>
bool configureAdapterLibraryState(
    AdapterLibraryState& state,
    int requestedMajor,
    SelectLibrary selectLibrary) {
  if (requestedMajor < 0 || requestedMajor > 255) {
    return false;
  }
  if (state.major != -1) {
    return state.handle != nullptr &&
        isAdapterLibraryMajorCompatible(state.major, requestedMajor);
  }

  const auto selection = selectLibrary(requestedMajor);
  if (selection.handle == nullptr) {
    return false;
  }
  state.handle = selection.handle;
  state.major = selection.major;
  state.loadAttempted = true;
  return true;
}

template <typename SelectLibrary>
void* getOrSelectAdapterLibrary(
    AdapterLibraryState& state,
    SelectLibrary selectLibrary) {
  if (state.loadAttempted) {
    return state.handle;
  }

  state.loadAttempted = true;
  const auto selection = selectLibrary(0);
  state.handle = selection.handle;
  state.major = selection.handle == nullptr ? 0 : selection.major;
  return state.handle;
}

} // namespace neuron
} // namespace backends
} // namespace executorch
