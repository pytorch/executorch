/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/named_data_map/pt2_archive_data_map.h>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

#include "miniz.h"

#include <nlohmann/json.hpp>
#include <string.h>
#include <unordered_map>

using json = nlohmann::json;

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

using executorch::aten::ScalarType;
using executorch::aten::string_view;
using executorch::ET_RUNTIME_NAMESPACE::TensorLayout;
using executorch::runtime::DataLoader;

using executorch::extension::MmapDataLoader;

// MZ_ZIP constants.
constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30;
constexpr int MZ_ZIP_LDH_FILENAME_LEN_OFS = 26;
constexpr int MZ_ZIP_LDH_EXTRA_LEN_OFS = 28;
constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIG = 0x04034b50;

// PT2Archive constants.
constexpr const char* WEIGHTS_DIR = "/data/weights/";
constexpr const char* WEIGHTS_CONFIG_FILE = "model_weights_config.json";
constexpr const char* CONSTANTS_DIR = "/data/constants/";
constexpr const char* CONSTANTS_CONFIG_FILE = "model_constants_config.json";

namespace {
ScalarType convert_pt2_to_et_scalartype(uint32_t dtype) {
  // PT2 serialization dtypes and ET dtypes are off by 1.
  // PT2: https://fburl.com/code/qjlmiifs (contains UNKNOWN at enum 0)
  // ET: https://fburl.com/code/gq30tizb (starts with BYTE at enum 0)
  return static_cast<ScalarType>(dtype - 1);
}

// Use to read miniz header info.
static int64_t read_le_16(uint8_t* buf) {
  return buf[0] + (buf[1] << 8);
}
} // namespace

namespace executorch {
namespace extension {

PT2ArchiveDataMap::~PT2ArchiveDataMap() {
  // Close zip archive resources.
  if (zip_archive_) {
    mz_zip_reader_end(zip_archive_.get());
  }
}

/*static*/ Error PT2ArchiveDataMap::parse_json(
    std::unique_ptr<mz_zip_archive>& zip_archive,
    const std::string& filename,
    std::unordered_map<std::string, std::string>& tensor_name_to_path,
    std::unordered_map<std::string, ConcreteTensorLayout>&
        tensor_name_to_layout) {
  /** JSON format (for information we care about) looks like this:
  "config": {
    "weight_name": {
      "path_name": "weight_0",
      "tensor_meta": {
        "dtype": <DTYPE>,
        "sizes": [{"as_int": <SIZE>}, {"as_int": <SIZE>}, ...],
        "strides": [{"as_int": <SIZE>}, {"as_int": <SIZE>}, ...],
      }
    }
  } */
  size_t uncomp_size = 0;
  void* buffer = mz_zip_reader_extract_file_to_heap(
      zip_archive.get(), filename.c_str(), &uncomp_size, 0);
  if (!buffer) {
    ET_LOG(Error, "Failed to extract file %s to heap", filename.c_str());
    mz_zip_reader_end(zip_archive.get());
    return Error::InvalidExternalData;
  }
  json json_config;
  try {
    std::string json_str(static_cast<const char*>(buffer), uncomp_size);
    // Parse JSON string.
    json_config = json::parse(json_str);
    ET_CHECK_OR_RETURN_ERROR(
        json_config.contains("config"),
        InvalidExternalData,
        "JSON config does not contain 'config' key; malformed archive file.");
    auto config = json_config["config"];
    for (auto& item : config.items()) {
      ET_CHECK_OR_RETURN_ERROR(
          item.value().contains("path_name") &&
              item.value().contains("tensor_meta"),
          InvalidExternalData,
          "JSON config does not contain 'path_name' and 'tensor_meta' keys for key %s",
          item.key().c_str());

      // Add tensor_name -> path_name mapping.
      tensor_name_to_path[item.key().c_str()] = item.value()["path_name"];

      // Add tensor_name -> tensor_meta mapping.
      auto tensor_meta = item.value()["tensor_meta"];
      ET_CHECK_OR_RETURN_ERROR(
          tensor_meta.contains("dtype") &&
              tensor_meta["dtype"].is_number_integer(),
          InvalidExternalData,
          "JSON config does not contain 'dtype' key for key %s",
          item.key().c_str());
      ET_CHECK_OR_RETURN_ERROR(
          tensor_meta.contains("sizes") && tensor_meta["sizes"].is_array(),
          InvalidExternalData,
          "JSON config does not contain 'sizes' key for key %s",
          item.key().c_str());
      ET_CHECK_OR_RETURN_ERROR(
          tensor_meta.contains("strides") && tensor_meta["strides"].is_array(),
          InvalidExternalData,
          "JSON config does not contain 'strides' key for key %s",
          item.key().c_str());
      ConcreteTensorLayout concrete_layout;
      concrete_layout.scalar_type =
          convert_pt2_to_et_scalartype(tensor_meta["dtype"].get<int>());
      int i = 0;
      for (const auto& size : tensor_meta["sizes"]) {
        concrete_layout.sizes.push_back(size["as_int"].get<int32_t>());
        // TODO: Calculate dim order from strides. Assume contiguous for now.
        concrete_layout.dim_order.push_back(i);
        ++i;
      }
      tensor_name_to_layout[item.key().c_str()] = std::move(concrete_layout);
    }
    free(buffer);
  } catch (const json::exception& e) {
    ET_LOG(Error, "Failed to parse JSON: %s", e.what());
    free(buffer);
    mz_zip_reader_end(zip_archive.get());
    return Error::InvalidExternalData;
  }
  return Error::Ok;
}

/*static*/ Result<PT2ArchiveDataMap> PT2ArchiveDataMap::load(
    const std::string& pt2_archive_file_path) {
  ET_LOG(
      Info, "Loading PT2ArchiveDataMap from %s", pt2_archive_file_path.c_str());
  auto zip_archive = std::make_unique<mz_zip_archive>();
  // Open zip archive to get json config data.
  memset(zip_archive.get(), 0, sizeof(mz_zip_archive));
  mz_bool status = mz_zip_reader_init_file(
      zip_archive.get(), pt2_archive_file_path.c_str(), 0);

  ET_CHECK_OR_RETURN_ERROR(
      status == 1,
      InvalidArgument,
      "Failed to open zip archive %s, status: %d",
      pt2_archive_file_path.c_str(),
      status);

  // Extract archive name.
  mz_uint n = mz_zip_reader_get_num_files(zip_archive.get());
  ET_CHECK_OR_RETURN_ERROR(
      n > 0, InvalidExternalData, "Archive does not contain any files");
  size_t name_size =
      mz_zip_reader_get_filename(zip_archive.get(), 0, nullptr, 0);
  std::string buf(name_size, '\0');
  mz_zip_reader_get_filename(zip_archive.get(), 0, &buf[0], name_size);
  auto pos = buf.find_first_of('/');
  ET_CHECK_OR_RETURN_ERROR(
      pos != std::string::npos,
      InvalidExternalData,
      "File in archive is not in a subdirectory");

  std::string archive_name = buf.substr(0, pos);

  // Set up data structures for tensor name -> {path, metadata}.
  std::unordered_map<std::string, std::string> tensor_name_to_path;
  std::unordered_map<std::string, ConcreteTensorLayout> tensor_name_to_layout;

  // Read model_weights.json file.
  std::string model_weights = archive_name + WEIGHTS_DIR + WEIGHTS_CONFIG_FILE;
  Error err = parse_json(
      zip_archive, model_weights, tensor_name_to_path, tensor_name_to_layout);
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      InvalidExternalData,
      "Failed to parse model weights json config");

  // Read model_constants.json file.
  std::string model_constants =
      archive_name + CONSTANTS_DIR + CONSTANTS_CONFIG_FILE;
  err = parse_json(
      zip_archive, model_constants, tensor_name_to_path, tensor_name_to_layout);
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      InvalidExternalData,
      "Failed to parse model constants json config");

  // Create data loader to wrap around zip archive.
  Result<MmapDataLoader> loader =
      MmapDataLoader::from(pt2_archive_file_path.c_str());
  ET_CHECK_OR_RETURN_ERROR(
      loader.ok(),
      InvalidArgument,
      "Loader failed to load with error: %zu",
      loader.error());

  std::unique_ptr<DataLoader> loader_ptr =
      std::make_unique<MmapDataLoader>(std::move(loader.get()));
  return PT2ArchiveDataMap(
      std::move(zip_archive),
      std::move(loader_ptr),
      std::move(archive_name),
      std::move(tensor_name_to_layout),
      std::move(tensor_name_to_path));
}

Result<const TensorLayout> PT2ArchiveDataMap::get_tensor_layout(
    string_view key) const {
  if (tensor_name_to_layout_.find(key.data()) == tensor_name_to_layout_.end()) {
    ET_LOG(Error, "Tensor layout not found for key %s", key.data());
    return Error::NotFound;
  }
  return tensor_name_to_layout_.at(key.data()).create_tensor_layout();
}

Result<FreeableBuffer> PT2ArchiveDataMap::get_data(string_view key) const {
  if (tensor_name_to_path_.find(key.data()) == tensor_name_to_path_.end()) {
    ET_LOG(Error, "Tensor data not found for key %s", key.data());
    return Error::NotFound;
  }

  // Load data from zip archive - see PyTorch equivalent:
  // https://www.internalfb.com/code/fbsource/[f25405534204]/fbcode/caffe2/caffe2/serialize/inline_container.cc?lines=614
  std::string file_path =
      archive_name_ + WEIGHTS_DIR + tensor_name_to_path_.at(key.data());
  int file_index = mz_zip_reader_locate_file(
      zip_archive_.get(), file_path.c_str(), nullptr, 0);

  mz_zip_archive_file_stat file_stat;
  if (!mz_zip_reader_file_stat(zip_archive_.get(), file_index, &file_stat)) {
    ET_LOG(Error, "Failed to get file stat for file '%s'\n", file_path.c_str());
    return Error::InvalidExternalData;
  }
  mz_uint64 file_size = file_stat.m_uncomp_size;
  mz_uint8 local_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
  if (mz_zip_read_archive_data(
          zip_archive_.get(),
          file_stat.m_local_header_ofs,
          local_header,
          MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE) {
    ET_LOG(Info, "Failed to read local header for '%s'\n", file_path.c_str());
    return Error::InvalidExternalData;
  }
  mz_uint32 sig = MZ_READ_LE32(local_header);
  if (sig != MZ_ZIP_LOCAL_DIR_HEADER_SIG) {
    ET_LOG(
        Info,
        "Invalid local header signature for '%s': 0x%08X\n",
        file_path.c_str(),
        sig);
    return Error::InvalidExternalData;
  }

  // Calculate offset.
  mz_uint16 filename_len =
      read_le_16(local_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
  mz_uint16 extra_len = read_le_16(local_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  mz_uint64 offset = file_stat.m_local_header_ofs +
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_len + extra_len;

  return loader_->load(
      offset,
      file_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
}

Error PT2ArchiveDataMap::load_data_into(
    ET_UNUSED string_view key,
    ET_UNUSED void* buffer,
    ET_UNUSED size_t size) const {
  return Error::NotImplemented;
}

Result<uint32_t> PT2ArchiveDataMap::get_num_keys() const {
  return tensor_name_to_path_.size();
}

Result<const char*> PT2ArchiveDataMap::get_key(uint32_t index) const {
  auto num_keys = get_num_keys().get();
  ET_CHECK_OR_RETURN_ERROR(
      index < num_keys,
      InvalidArgument,
      "Index %u out of range of size %u",
      index,
      num_keys);
  int i = 0;
  for (const auto& item : tensor_name_to_path_) {
    if (i == index) {
      return item.first.c_str();
    }
    ++i;
  }
  // Should not reach here.
  return Error::Internal;
}

} // namespace extension
} // namespace executorch
