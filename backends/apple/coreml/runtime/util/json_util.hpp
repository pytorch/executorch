//
//  json_util.hpp
//  util
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <iostream>
#include <optional>
#include <sstream>

namespace executorchcoreml {
namespace json {
/// Reads and returns the first json object from the stream.
///
/// The returned string is not guaranteed to be a valid json object, the caller must
/// use a JSON parser to parse the returned string into a json object.
///
///
/// @param stream  The stream to read from.
/// @param max_bytes_to_read  The maximum bytes that can be read from the stream.
/// @retval The json object string or `nullopt` if there is no json object in the stream.
std::optional<std::string> read_object_from_stream(std::istream& stream,
                                                   size_t max_bytes_to_read = 10 * 1024 * 1024);

} // namespace json
} // namespace executorchcoreml
