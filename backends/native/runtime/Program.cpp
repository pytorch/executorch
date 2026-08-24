// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Program.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {

namespace {
// Minimum bytes for a FlatBuffer carrying a file identifier: a 4-byte root
// offset plus the 4-byte identifier.
constexpr size_t kMinBufferSize = 8;
} // namespace

Program Program::load(const void* data, size_t size) {
  if (data == nullptr || size < kMinBufferSize) {
    throw std::runtime_error("native program: buffer is null or too small");
  }

  const uint8_t* begin = static_cast<const uint8_t*>(data);
  std::vector<uint8_t> bytes(begin, begin + size);

  if (!::native_backend::ProgramBufferHasIdentifier(bytes.data())) {
    throw std::runtime_error(
        "native program: bad FlatBuffer file identifier (expected 'NPTG')");
  }

  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!::native_backend::VerifyProgramBuffer(verifier)) {
    throw std::runtime_error("native program: FlatBuffer verification failed");
  }

  const ::native_backend::Program* program_fb =
      ::native_backend::GetProgram(bytes.data());
  return Program(std::move(bytes), program_fb);
}

size_t Program::num_methods() const {
  const auto* methods = program_fb_->methods();
  return methods == nullptr ? 0 : methods->size();
}

} // namespace ptn
