// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/Samples.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/deserialize/Json.h>
#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>

namespace ptn {
namespace {

bool is_ssa_name(std::string_view name) {
  return !name.empty() && !std::isdigit(static_cast<unsigned char>(name[0])) &&
      std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
      });
}

bool is_digits(std::string_view text) {
  return !text.empty() && (text == "0" || text[0] != '0') &&
      std::all_of(text.begin(), text.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
      });
}

// "MAJOR.MINOR" with major 1; a newer minor only adds fields.
bool is_supported_version(const Json& version) {
  if (!version.is_string()) {
    return false;
  }
  const std::string_view text = version.get_ref<const std::string&>();
  const size_t dot = text.find('.');
  return dot != std::string_view::npos && text.substr(0, dot) == "1" &&
      is_digits(text.substr(dot + 1));
}

class SetDecoder {
 private:
  std::string where_;
  std::vector<uint8_t> blob_;
  SafeTensorsReader tensors_;
  size_t data_start_ = 0;
  std::unordered_set<std::string> referenced_;

 public:
  SetDecoder(std::string where, std::vector<uint8_t> blob)
      : where_(std::move(where)),
        blob_(std::move(blob)),
        tensors_(SafeTensorsReader::open(blob_)) {
    data_start_ = SafeTensorsReader::kLengthPrefixSize +
        SafeTensorsReader::header_size(ByteSpan(blob_).first(
            SafeTensorsReader::kLengthPrefixSize));
  }

  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error(where_ + ": " + message);
  }

  std::optional<Tolerance> tol(
      const Json& object,
      std::optional<Tolerance> fallback) const {
    const auto it = object.find("tol");
    if (it == object.end()) {
      return fallback;
    }
    if (!it->is_array() || it->size() != 2) {
      fail("tol must be [atol, rtol]");
    }
    for (const Json& t : *it) {
      if (!t.is_number() || !std::isfinite(t.get<double>()) ||
          t.get<double>() < 0) {
        fail("tol must be two finite numbers >= 0");
      }
    }
    return Tolerance{(*it)[0].get<double>(), (*it)[1].get<double>()};
  }

  SampleTensor tensor(const Json& value, const std::string& key) {
    if (!value.is_string()) {
      fail(key + " must be a tensor; scalar and list values are not supported");
    }
    const std::string& name = value.get_ref<const std::string&>();
    if (name != key) {
      fail("key " + name + " must be " + key);
    }
    const TensorEntry* entry = tensors_.find(name);
    if (entry == nullptr) {
      fail("no tensor named " + name);
    }
    referenced_.insert(name);
    const uint8_t* begin = blob_.data() + data_start_ + entry->offset;
    return {entry->dtype, entry->sizes, {begin, begin + entry->nbytes}};
  }

  std::vector<SampleTensor> tensors(
      const Json& values,
      const std::string& key) {
    if (!values.is_array()) {
      fail(key + " must be a list");
    }
    std::vector<SampleTensor> result;
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      result.push_back(tensor(values[i], key + "[" + std::to_string(i) + "]"));
    }
    return result;
  }

  SampleCall call(
      const Json& entry,
      const std::string& at,
      std::optional<Tolerance> fallback) {
    if (!entry.is_object() || !entry.contains("in")) {
      fail(at + " must be an object with an 'in' list");
    }
    SampleCall result;
    result.inputs = tensors(entry["in"], at + ".in");
    if (entry.contains("out")) {
      result.outputs = tensors(entry["out"], at + ".out");
    }
    if (entry.contains("intermediates")) {
      const Json& intermediates = entry["intermediates"];
      if (!intermediates.is_object()) {
        fail(at + ".intermediates must be an object");
      }
      for (const auto& [name, key] : intermediates.items()) {
        if (!is_ssa_name(name)) {
          fail("bad intermediate " + name + " in " + at);
        }
        result.intermediates.emplace(
            name, tensor(key, at + ".intermediates." + name));
      }
    }
    result.tol = tol(entry, fallback);
    return result;
  }

  void check_all_referenced() const {
    for (const std::string& name : tensors_.names()) {
      if (!referenced_.contains(name)) {
        fail("unreferenced tensor " + name);
      }
    }
  }
};

} // namespace

SampleSet read_sample_set(
    const ZipReader& zip,
    const std::string& method,
    const std::string& set) {
  const std::string prefix = "samples/" + method + "/" + set + "/";
  const std::string where = "samples: " + prefix;
  if (!zip.member_size(prefix + "index.json") ||
      !zip.member_size(prefix + "data.safetensors")) {
    throw std::runtime_error(
        where + ": missing index.json or data.safetensors");
  }
  const std::vector<uint8_t> index_bytes = zip.read(prefix + "index.json");
  Json index;
  try {
    index = parse_json(std::string_view(
        reinterpret_cast<const char*>(index_bytes.data()), index_bytes.size()));
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        where + ": invalid index.json: " + std::string(error.what()));
  }
  SetDecoder decoder(where, zip.read(prefix + "data.safetensors"));

  if (!index.is_object()) {
    decoder.fail("index.json must contain a JSON object");
  }
  const auto version = index.find("version");
  if (version == index.end() || !is_supported_version(*version)) {
    decoder.fail("unsupported version, expected 1.0");
  }
  SampleSet result;
  for (const auto& [field, value] :
       {std::pair{"description", &result.description},
        std::pair{"source", &result.source}}) {
    if (index.contains(field)) {
      if (!index[field].is_string()) {
        decoder.fail(std::string(field) + " must be a string");
      }
      *value = index[field].get<std::string>();
    }
  }
  const auto entries = index.find("samples");
  if (entries == index.end() || !entries->is_array() || entries->empty()) {
    decoder.fail("samples must be a non-empty list");
  }
  const std::optional<Tolerance> set_tol = decoder.tol(index, std::nullopt);
  for (size_t i = 0; i < entries->size(); ++i) {
    const Json& entry = (*entries)[i];
    const std::string at = "samples[" + std::to_string(i) + "]";
    if (!entry.is_object() || !entry.contains("calls") ||
        !entry["calls"].is_array() || entry["calls"].empty()) {
      decoder.fail(at + " must be an object with a non-empty 'calls' list");
    }
    const std::optional<Tolerance> sample_tol = decoder.tol(entry, set_tol);
    Sample& sample = result.samples.emplace_back();
    const Json& calls = entry["calls"];
    for (size_t j = 0; j < calls.size(); ++j) {
      sample.calls.push_back(decoder.call(
          calls[j], at + ".calls[" + std::to_string(j) + "]", sample_tol));
    }
  }
  decoder.check_all_referenced();
  return result;
}

} // namespace ptn
