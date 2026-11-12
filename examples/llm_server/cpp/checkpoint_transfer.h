/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/serving/serving_runtime.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <limits>
#include <stdexcept>

namespace executorch::examples::llm_server::checkpoint_transfer {
using Json = nlohmann::json;
namespace cache = extension::llm::cache;
namespace serving = extension::llm::serving;

// Temporary local IPC staging, not a durable checkpoint format. Python owns
// the file, integrity checks, compatibility identity and atomic publication.
class BufferFile {
 public:
  BufferFile(const std::string& path, bool writing, uint64_t offset = 0)
      : offset_(offset) {
    fd_ = open(
        path.c_str(),
        (writing ? O_WRONLY : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat info {};
    if (fd_ < 0 || fstat(fd_, &info) || !S_ISREG(info.st_mode) ||
        info.st_size < 0 || offset > static_cast<uint64_t>(info.st_size) ||
        (writing &&
         (info.st_uid != getuid() || info.st_nlink != 1 ||
          info.st_size != 0))) {
      if (fd_ >= 0)
        close(fd_);
      throw std::invalid_argument("invalid checkpoint staging file");
    }
    size_ = static_cast<uint64_t>(info.st_size);
  }
  ~BufferFile() {
    close(fd_);
  }
  BufferFile(const BufferFile&) = delete;
  BufferFile& operator=(const BufferFile&) = delete;

  Json put(const std::vector<uint8_t>& bytes) {
    const auto begin = offset_;
    size_t done = 0;
    while (done < bytes.size()) {
      const auto count = pwrite(
          fd_,
          bytes.data() + done,
          std::min<size_t>(bytes.size() - done, 8 * 1024 * 1024),
          offset_);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        throw std::runtime_error("checkpoint staging write failed");
      done += count;
      offset_ += count;
    }
    return Json::array({begin, bytes.size()});
  }

  std::vector<uint8_t> get(const Json& descriptor, uint64_t base) {
    if (!descriptor.is_array() || descriptor.size() != 2 ||
        !descriptor[0].is_number_unsigned() ||
        !descriptor[1].is_number_unsigned()) {
      throw std::invalid_argument("invalid checkpoint buffer descriptor");
    }
    const auto at = descriptor[0].get<uint64_t>();
    const auto bytes = descriptor[1].get<uint64_t>();
    if (base > size_ || at > size_ - base || bytes > size_ - base - at ||
        base + at != offset_ || bytes > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument(
          "checkpoint buffers are truncated or out of order");
    }
    std::vector<uint8_t> result(bytes);
    size_t done = 0;
    while (done < result.size()) {
      const auto count = pread(
          fd_,
          result.data() + done,
          std::min<size_t>(result.size() - done, 8 * 1024 * 1024),
          offset_);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        throw std::runtime_error("checkpoint staging read failed");
      done += count;
      offset_ += count;
    }
    return result;
  }
  uint64_t offset() const {
    return offset_;
  }

 private:
  int fd_;
  uint64_t size_ = 0, offset_ = 0;
};

inline Json export_buffers(
    const serving::SessionCheckpoint& checkpoint,
    const std::string& path) {
  BufferFile file(path, true);
  const auto& state = checkpoint.state;
  Json manifest = {
      {"format", state.format},
      {"position", state.position},
      {"hidden_row_bytes", state.hidden_row_bytes},
      {"tokens", checkpoint.tokens},
      {"caches", Json::array()}};
  for (const auto& cache : state.caches) {
    Json record = {
        {"position", cache.position},
        {"dtype", cache.dtype},
        {"dflash_block_size",
         cache.dflash_block_size ? Json(*cache.dflash_block_size)
                                 : Json(nullptr)},
        {"layers", Json::array()}};
    for (const auto& layer : cache.layers) {
      record["layers"].push_back(
          {{"begin", layer.begin},
           {"heads", layer.heads},
           {"head_dim", layer.head_dim},
           {"window", layer.window},
           {"key", file.put(layer.key)},
           {"value", file.put(layer.value)}});
    }
    manifest["caches"].push_back(std::move(record));
  }
  manifest["hidden"] = file.put(state.hidden);
  manifest["data_bytes"] = file.offset();
  return manifest;
}

inline int integer(const Json& value) {
  if (!value.is_number_integer() || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("invalid checkpoint integer");
  }
  return value.get<int>();
}

inline std::shared_ptr<serving::SessionCheckpoint>
import_buffers(const Json& manifest, const std::string& path, uint64_t offset) {
  BufferFile file(path, false, offset);
  auto checkpoint = std::make_shared<serving::SessionCheckpoint>();
  auto& state = checkpoint->state;
  state.format = manifest.at("format").get<std::string>();
  state.position = integer(manifest.at("position"));
  state.hidden_row_bytes = integer(manifest.at("hidden_row_bytes"));
  const auto& tokens = manifest.at("tokens");
  if (!tokens.is_array() ||
      tokens.size() != static_cast<size_t>(state.position)) {
    throw std::invalid_argument("invalid checkpoint token history");
  }
  for (const auto& token : tokens)
    checkpoint->tokens.push_back(integer(token));
  const auto& caches = manifest.at("caches");
  if (!caches.is_array() || caches.empty() || caches.size() > 2) {
    throw std::invalid_argument("invalid checkpoint cache count");
  }
  for (const auto& record : caches) {
    cache::SequenceCheckpoint cache;
    cache.position = integer(record.at("position"));
    cache.dtype = integer(record.at("dtype"));
    if (!record.at("dflash_block_size").is_null())
      cache.dflash_block_size = integer(record.at("dflash_block_size"));
    const auto& layers = record.at("layers");
    if (!layers.is_array() || layers.empty() || layers.size() > 1024) {
      throw std::invalid_argument("invalid checkpoint layer count");
    }
    for (const auto& layer_record : layers) {
      cache::LayerCheckpoint layer;
      layer.begin = integer(layer_record.at("begin"));
      layer.heads = integer(layer_record.at("heads"));
      layer.head_dim = integer(layer_record.at("head_dim"));
      layer.window = integer(layer_record.at("window"));
      layer.key = file.get(layer_record.at("key"), offset);
      layer.value = file.get(layer_record.at("value"), offset);
      cache.layers.push_back(std::move(layer));
    }
    state.caches.push_back(std::move(cache));
  }
  state.hidden = file.get(manifest.at("hidden"), offset);
  if (file.offset() - offset != manifest.at("data_bytes").get<uint64_t>()) {
    throw std::invalid_argument("checkpoint payload size mismatch");
  }
  return checkpoint;
}
} // namespace executorch::examples::llm_server::checkpoint_transfer
