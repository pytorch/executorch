//
//  json_util.cpp
//  util
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "json_util.hpp"

#include <string>
#include <vector>

namespace {

struct JSONParseState {
    int n_open_braces = 0;
    std::string json_object;
    bool reading_string = false;
};

bool is_quote_escaped(const std::vector<char>& buffer, size_t offset) {
    return offset >= 1 && buffer[offset - 1] == '\\';
}

void process_quotes(const std::vector<char>& buffer, JSONParseState& state, size_t offset) {
    // If we are reading a string and the quote is escaped then we don't mark it as the end of string.
    if (state.reading_string && is_quote_escaped(buffer, offset)) {
        return;
    }
    state.reading_string = !state.reading_string;
}

// Foundation has no streaming parser, so we try to read the top json object.
// At this stage we don't care about the validity of json, the parser would
// anyways fail if the json object is not valid.
void read_object_from_buffer(std::vector<char>& buffer, JSONParseState& state) {
    size_t offset = 0;
    for (; offset < buffer.size() && state.n_open_braces > 0; offset++) {
        switch (buffer[offset]) {
            case '}': {
                // If the close brace is inside a string then then ignore it.
                state.n_open_braces -= (state.reading_string ? 0 : 1);
                break;
            }
            case '{': {
                // If the open brace is inside a string then ignore it.
                state.n_open_braces += (state.reading_string ? 0 : 1);
                break;
            }
            case '"': {
                process_quotes(buffer, state, offset);
                break;
            }
            default: {
                break;
            }
        }
    }

    if (offset > 0) {
        state.json_object.append(buffer.begin(), buffer.begin() + offset);
    }
}
}

namespace executorchcoreml {
namespace json {

std::optional<std::string> read_object_from_stream(std::istream& stream, size_t max_bytes_to_read) {
    static constexpr size_t buffer_size = 512;
    JSONParseState state;
    char ch;
    // Ignore 0, 0 is added for padding.
    do {
        stream >> ch;
    } while (stream.good() && static_cast<uint8_t>(ch) == 0);
    // The first character must be an opening brace.
    if (ch != '{') {
        return std::optional<std::string>();
    }
    state.json_object += ch;
    state.n_open_braces = 1;

    std::vector<char> buffer;
    while (stream.good() && state.n_open_braces > 0 && state.json_object.size() < max_bytes_to_read) {
        buffer.resize(buffer_size, '\0');
        stream.read(buffer.data(), buffer_size);
        read_object_from_buffer(buffer, state);
        buffer.clear();
    }

    return state.n_open_braces == 0 ? state.json_object : std::optional<std::string>();
}

} // namespace json
} // namespace executorchcoreml
