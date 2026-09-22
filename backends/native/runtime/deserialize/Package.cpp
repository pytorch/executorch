// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/Package.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>
#include <string_view>

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Json.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>

namespace ptn {
namespace {

// Reserved by safetensors, so it can never name a constant.
constexpr std::string_view kMetadataKey = "__metadata__";
std::atomic<uint64_t> next_package_id{1};

std::unordered_map<std::string, std::string> parse_aliases(
    ByteSpan member,
    const SafeTensorsReader& tensors) {
  Json doc;
  try {
    doc = parse_json(std::string_view(
        reinterpret_cast<const char*>(member.data()), member.size()));
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        "package: invalid aliases.json: " + std::string(error.what()));
  }
  if (!doc.is_object()) {
    throw std::runtime_error("package: aliases.json is not a JSON object");
  }
  if (doc.size() > detail::kMaxAliasCount) {
    throw ResourceLimitError("package: alias count exceeds limit");
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
    aliases.emplace(key, owner);
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

Package::Package()
    : id_(next_package_id.fetch_add(1, std::memory_order_relaxed)) {}

Package Package::load(OwnedBytes bytes) {
  Package out;
  out.archive_bytes_ = std::move(bytes);
  if (out.archive_bytes_.span().size() > detail::kMaxPackageBytes) {
    throw ResourceLimitError("package: image exceeds size limit");
  }
  out.zip_ = ZipReader::open(out.archive_bytes_.span());
  out.load_metadata();
  return out;
}

Package Package::load(const std::string& path) {
  Package out;
  out.zip_ = ZipReader::open(path);
  out.load_metadata();
  return out;
}

Package& Package::operator=(Package&& other) noexcept {
  if (this != &other) {
    zip_.reset();
    id_ = other.id_;
    archive_bytes_ = std::move(other.archive_bytes_);
    zip_ = std::move(other.zip_);
    program_ = std::move(other.program_);
    tensors_ = std::move(other.tensors_);
    tensor_data_offset_ = other.tensor_data_offset_;
    aliases_ = std::move(other.aliases_);
  }
  return *this;
}

void Package::load_metadata() {
  const std::optional<size_t> program_size = zip_->member_size(kProgramEntry);
  if (!program_size) {
    throw std::runtime_error(
        std::string("package: missing required member ") + kProgramEntry);
  }
  if (*program_size > detail::kMaxProgramBytes) {
    throw ResourceLimitError("package: program exceeds size limit");
  }
  program_ = zip_->read(kProgramEntry);

  // Absent whenever the program references no constants, which is normal for a
  // graph over user inputs alone.
  const std::optional<size_t> tensor_size =
      zip_->member_size(kSafeTensorsEntry);
  if (tensor_size) {
    std::array<uint8_t, SafeTensorsReader::kLengthPrefixSize> prefix{};
    if (*tensor_size < prefix.size()) {
      throw std::runtime_error(
          "package: safetensors member is shorter than its length prefix");
    }
    zip_->read_into(kSafeTensorsEntry, 0, MutableByteSpan(prefix));
    const size_t header_size = SafeTensorsReader::header_size(prefix);
    if (header_size > detail::kMaxJsonBytes) {
      throw ResourceLimitError(
          "package: safetensors header exceeds size limit");
    }
    if (header_size > *tensor_size - prefix.size()) {
      throw std::runtime_error(
          "package: safetensors header exceeds its zip member");
    }
    std::vector<uint8_t> header(header_size);
    zip_->read_into(kSafeTensorsEntry, prefix.size(), MutableByteSpan(header));
    tensor_data_offset_ = prefix.size() + header_size;
    tensors_ = SafeTensorsReader::open_header(
        ByteSpan(header), *tensor_size - tensor_data_offset_);
  }

  const std::optional<size_t> aliases_size = zip_->member_size(kAliasesEntry);
  if (aliases_size) {
    if (!tensors_) {
      throw std::runtime_error(
          std::string("package: has ") + kAliasesEntry + " but no " +
          kSafeTensorsEntry);
    }
    if (*aliases_size > detail::kMaxJsonBytes) {
      throw ResourceLimitError("package: aliases exceed size limit");
    }
    const std::vector<uint8_t> aliases = zip_->read(kAliasesEntry);
    aliases_ = parse_aliases(ByteSpan(aliases), *tensors_);
  }
}

std::optional<ConstantInfo> Package::constant_info(
    const std::string& key) const {
  if (!tensors_) {
    return std::nullopt;
  }
  const auto alias = aliases_.find(key);
  const std::string& owner = alias == aliases_.end() ? key : alias->second;

  const TensorEntry* entry = tensors_->find(owner);
  if (entry == nullptr) {
    return std::nullopt;
  }

  ConstantInfo out;
  out.package_id = id_;
  out.dtype = entry->dtype;
  out.sizes = &entry->sizes;
  out.nbytes = entry->nbytes;
  out.owner = owner;
  return out;
}

std::optional<OwnedBytes> Package::acquire_constant(
    const std::string& key) const {
  const std::optional<ConstantInfo> info = constant_info(key);
  if (!info) {
    return std::nullopt;
  }
  std::vector<uint8_t> bytes(info->nbytes);
  load_constant_into(key, MutableByteSpan(bytes));
  return OwnedBytes::from_vector(std::move(bytes));
}

bool Package::load_constant_into(
    const std::string& key,
    MutableByteSpan destination) const {
  const std::optional<ConstantInfo> info = constant_info(key);
  if (!info) {
    return false;
  }
  if (destination.size() != info->nbytes) {
    throw std::runtime_error(
        "package: destination for '" + key + "' has " +
        std::to_string(destination.size()) + " bytes; expected " +
        std::to_string(info->nbytes));
  }
  const TensorEntry* entry = tensors_->find(info->owner);
  zip_->read_into(
      kSafeTensorsEntry, tensor_data_offset_ + entry->offset, destination);
  return true;
}

void Package::verify() const {
  zip_->verify(kProgramEntry);
  if (tensors_) {
    zip_->verify(kSafeTensorsEntry);
  }
  if (zip_->member_size(kAliasesEntry)) {
    zip_->verify(kAliasesEntry);
  }
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
