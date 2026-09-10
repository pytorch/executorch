// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/Package.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <executorch/backends/native/runtime/deserialize/Json.h>

namespace ptn {
namespace {

// Reserved by safetensors, so it can never name a constant.
constexpr std::string_view kMetadataKey = "__metadata__";

std::unordered_map<std::string, std::string> parse_aliases(
    ByteSpan member,
    const SafeTensorsReader& tensors) {
  Json doc;
  try {
    doc = Json::parse(std::string_view(
        reinterpret_cast<const char*>(member.data()), member.size()));
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        "package: invalid aliases.json: " + std::string(error.what()));
  }
  if (!doc.is_object()) {
    throw std::runtime_error("package: aliases.json is not a JSON object");
  }

  std::unordered_map<std::string, std::string> aliases;
  for (auto entry = doc.begin(); entry != doc.end(); ++entry) {
    const std::string& key = entry.key();
    if (!entry.value().is_string()) {
      throw std::runtime_error(
          "package: alias '" + key + "' does not name a string owner");
    }
    const std::string& owner = entry.value().get_ref<const std::string&>();
    if (key == kMetadataKey) {
      throw std::runtime_error(
          "package: alias key is reserved by safetensors: " + key);
    }
    // An owner is always a real safetensors entry and an alias is never one, so
    // resolution stays a single lookup. Enforce both rather than trusting it.
    if (tensors.find(owner) == nullptr) {
      std::string message = "package: alias '";
      message += key;
      message += "' names owner '";
      message += owner;
      message += "', which has no safetensors entry";
      throw std::runtime_error(message);
    }
    if (tensors.find(key) != nullptr) {
      throw std::runtime_error(
          "package: '" + key + "' is both a safetensors owner and an alias");
    }
    if (!aliases.emplace(key, owner).second) {
      throw std::runtime_error("package: duplicate alias key: " + key);
    }
  }
  return aliases;
}

} // namespace

bool Package::looks_like_package(ByteSpan bytes) {
  // Every zip record signature begins "PK"; a bare .ptg starts with a
  // flatbuffer root offset followed by "NPTG" at offset 4, so this cannot
  // collide.
  return bytes.size() >= 2 && bytes[0] == 'P' && bytes[1] == 'K';
}

Package Package::load(std::vector<uint8_t> bytes) {
  Package out;
  out.bytes_ = std::move(bytes);
  const ByteSpan image{out.bytes_.data(), out.bytes_.size()};

  out.zip_ = ZipReader::open(image);

  if (!out.zip_->member_size(kProgramEntry)) {
    throw std::runtime_error(
        std::string("package: missing required member ") + kProgramEntry);
  }
  out.program_ = out.zip_->read(kProgramEntry);

  // Absent whenever the program references no constants, which is normal for a
  // graph over user inputs alone.
  if (out.zip_->member_size(kSafeTensorsEntry)) {
    out.tensor_bytes_ = out.zip_->read(kSafeTensorsEntry);
    out.tensors_ = SafeTensorsReader::open(ByteSpan(out.tensor_bytes_));
  }

  if (out.zip_->member_size(kAliasesEntry)) {
    if (!out.tensors_) {
      throw std::runtime_error(
          std::string("package: has ") + kAliasesEntry + " but no " +
          kSafeTensorsEntry);
    }
    const std::vector<uint8_t> aliases = out.zip_->read(kAliasesEntry);
    out.aliases_ = parse_aliases(ByteSpan(aliases), *out.tensors_);
  }

  return out;
}

Package Package::load_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("package: cannot open " + path);
  }
  const std::streamsize size = file.tellg();
  if (size < 0) {
    throw std::runtime_error("package: cannot size " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("package: cannot read " + path);
  }
  return load(std::move(bytes));
}

std::optional<Constant> Package::constant(const std::string& key) const {
  if (!tensors_) {
    return std::nullopt;
  }
  const auto alias = aliases_.find(key);
  const std::string& owner = alias == aliases_.end() ? key : alias->second;

  const TensorEntry* entry = tensors_->find(owner);
  if (entry == nullptr) {
    return std::nullopt;
  }

  Constant out;
  out.dtype = entry->dtype;
  out.sizes = &entry->sizes;
  out.bytes = tensors_->bytes(*entry);
  out.owner = owner;
  return out;
}

std::vector<std::string> Package::keys() const {
  std::vector<std::string> out;
  if (tensors_) {
    out = tensors_->names();
  }
  for (const auto& alias : aliases_) {
    out.push_back(alias.first);
  }
  std::ranges::sort(out);
  return out;
}

const std::vector<std::string>& Package::owner_keys() const {
  static const std::vector<std::string> kNone;
  return tensors_ ? tensors_->names() : kNone;
}

size_t Package::constant_bytes() const {
  return tensors_ ? tensors_->total_bytes() : 0;
}

} // namespace ptn
