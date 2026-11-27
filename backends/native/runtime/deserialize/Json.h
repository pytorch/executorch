// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>

namespace ptn {

using Json = nlohmann::ordered_json;

inline Json parse_json(std::string_view text) {
  if (text.size() > detail::kMaxJsonBytes) {
    throw ResourceLimitError("json: document exceeds size limit");
  }

  size_t value_count = 0;
  std::vector<std::unordered_set<std::string>> object_keys;
  const Json::parser_callback_t callback =
      [&](int depth, Json::parse_event_t event, Json& parsed) {
        if (depth > detail::kMaxJsonDepth) {
          throw ResourceLimitError("json: document exceeds depth limit");
        }
        if (event == Json::parse_event_t::object_start ||
            event == Json::parse_event_t::array_start ||
            event == Json::parse_event_t::value) {
          if (value_count >= detail::kMaxJsonValues) {
            throw ResourceLimitError("json: document exceeds value limit");
          }
          ++value_count;
        }
        if (event == Json::parse_event_t::object_start) {
          // nlohmann reports object_start at its parent's depth, while key uses
          // the object's depth.
          const size_t object_depth = static_cast<size_t>(depth) + 1;
          if (object_keys.size() <= object_depth) {
            object_keys.resize(object_depth + 1);
          }
          object_keys[object_depth].clear();
        } else if (event == Json::parse_event_t::key) {
          const size_t object_depth = static_cast<size_t>(depth);
          if (object_keys.size() <= object_depth) {
            object_keys.resize(object_depth + 1);
          }
          const std::string& key = parsed.get_ref<const std::string&>();
          if (!object_keys[object_depth].insert(key).second) {
            throw std::runtime_error("json: duplicate object key: " + key);
          }
        }
        return true;
      };
  return Json::parse(text, callback);
}

} // namespace ptn
