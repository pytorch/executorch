// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Program.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {

namespace {
// Minimum bytes for a FlatBuffer carrying a file identifier: a 4-byte root
// offset plus the 4-byte identifier.
constexpr size_t kMinBufferSize = 8;
constexpr uint32_t kSupportedMajor = 1;
constexpr uint32_t kSupportedMinor = 0;

uint32_t parse_version_component(std::string_view component) {
  if (component.empty()) {
    throw std::runtime_error("native program: malformed schema version");
  }
  uint32_t value = 0;
  for (const char digit : component) {
    if (digit < '0' || digit > '9') {
      throw std::runtime_error("native program: malformed schema version");
    }
    const uint32_t next = static_cast<uint32_t>(digit - '0');
    if (value > (std::numeric_limits<uint32_t>::max() - next) / 10) {
      throw std::runtime_error("native program: schema version overflows");
    }
    value = value * 10 + next;
  }
  return value;
}

ProgramVersion parse_version(const flatbuffers::String* serialized) {
  if (serialized == nullptr) {
    throw std::runtime_error("native program: schema version is missing");
  }
  const std::string_view text(serialized->c_str(), serialized->size());
  const size_t separator = text.find('.');
  if (separator == std::string_view::npos) {
    return ProgramVersion{parse_version_component(text), /*minor=*/0};
  }
  if (text.find('.', separator + 1) != std::string_view::npos) {
    throw std::runtime_error("native program: malformed schema version");
  }
  return ProgramVersion{
      parse_version_component(text.substr(/*pos=*/0, separator)),
      parse_version_component(text.substr(separator + 1))};
}
} // namespace

Program::Program(std::vector<uint8_t> bytes, const fbs::Program* program_fb)
    : bytes_(std::move(bytes)),
      program_fb_(program_fb),
      version_(parse_version(program_fb_->version())) {
  if (version_.major != kSupportedMajor || version_.minor > kSupportedMinor) {
    throw UnsupportedVersionError(
        "native program: unsupported schema version " +
        std::to_string(version_.major) + "." + std::to_string(version_.minor));
  }
}

Program::Program(Program&& other) noexcept
    : bytes_(std::move(other.bytes_)),
      program_fb_(other.program_fb_),
      version_(other.version_),
      method_cache_(std::move(other.method_cache_)) {
  other.program_fb_ = nullptr;
  other.version_ = {};
}

Program& Program::operator=(Program&& other) noexcept {
  if (this != &other) {
    bytes_ = std::move(other.bytes_);
    program_fb_ = other.program_fb_;
    version_ = other.version_;
    method_cache_ = std::move(other.method_cache_);
    other.program_fb_ = nullptr;
    other.version_ = {};
  }
  return *this;
}

Program Program::load(const void* data, size_t size) {
  if (data == nullptr || size < kMinBufferSize) {
    throw std::runtime_error("native program: buffer is null or too small");
  }

  const uint8_t* begin = static_cast<const uint8_t*>(data);
  std::vector<uint8_t> bytes(begin, begin + size);

  if (!fbs::ProgramBufferHasIdentifier(bytes.data())) {
    throw std::runtime_error(
        "native program: bad FlatBuffer file identifier (expected 'NPTG')");
  }

  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!fbs::VerifyProgramBuffer(verifier)) {
    throw std::runtime_error("native program: FlatBuffer verification failed");
  }

  const fbs::Program* program_fb = fbs::GetProgram(bytes.data());
  // Both accessors below are schema-required, so successful verification
  // guarantees that they are non-null.
  std::unordered_set<std::string> method_names;
  for (const fbs::Method* method : *program_fb->methods()) {
    const std::string name = method->name()->str();
    if (name.empty()) {
      throw std::runtime_error("native program: method name is empty");
    }
    if (!method_names.insert(name).second) {
      throw std::runtime_error(
          "native program: duplicate method name '" + name + "'");
    }
  }
  return Program(std::move(bytes), program_fb);
}

size_t Program::num_methods() const {
  if (program_fb_ == nullptr) {
    return 0;
  }
  const auto* methods = program_fb_->methods();
  return methods == nullptr ? 0 : methods->size();
}

std::vector<std::string> Program::method_names() const {
  std::vector<std::string> names;
  if (program_fb_ == nullptr) {
    return names;
  }
  const auto* methods = program_fb_->methods();
  if (methods != nullptr) {
    names.reserve(methods->size());
    for (flatbuffers::uoffset_t i = 0; i < methods->size(); ++i) {
      const auto* nm = methods->Get(i)->name();
      names.push_back(nm != nullptr ? nm->str() : std::string());
    }
  }
  return names;
}

} // namespace ptn
