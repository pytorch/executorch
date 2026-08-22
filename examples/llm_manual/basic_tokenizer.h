/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class BasicTokenizer {
 public:
  explicit BasicTokenizer(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file) {
      std::cerr << "Unable to open file " << file_path << "\n";
      exit(9);
    }
    std::string str(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    size_t i = 0u;
    i = consume_whitespace(str, i);
    i = expect(str, i, '{');

    while (i < str.size() && str[i] != '}') {
      i = consume_field(str, i);
    }

    // Build decode map as inverse of encode.
    for (auto& i : encode_) {
      decode_[i.second] = i.first;
    }
  }

  std::vector<int64_t> encode(const std::string& prompt) {
    std::vector<std::string> words = parse_prompt(prompt);
    std::vector<int64_t> result;
    for (auto word : words) {
      result.push_back(encode_[word]);
    }
    return result;
  }

  std::string decode(const std::vector<int64_t>& indices) {
    std::string result;
    for (const auto& index : indices) {
      result += decode_[index];
    }
    return result;
  }

 private:
  std::unordered_map<std::string, int64_t> encode_;
  std::unordered_map<int64_t, std::string> decode_;

  // Advance the input string index until a non-whitespace character is found
  // or it reaches the end of string.
  size_t consume_whitespace(const std::string& data, size_t i) {
    while (i < data.size() && std::isspace(data[i])) {
      i++;
    }

    return i;
  }

  // Consumes an JSON field of the form
  //  "str": id,
  size_t consume_field(const std::string& data, size_t i) {
    i = consume_whitespace(data, i);

    // Parse the key literal.
    i = expect(data, i, '"');

    auto in_escape = false;
    std::string key = "";
    while (i < data.size()) {
      if (in_escape) {
        key += data[i];
        i++;
        in_escape = false;
      } else { // !in_escape
        if (data[i] == '"') { // End of string literal
          i++;
          break;
        } else if (data[i] == '\\') { // Escaped code point
          in_escape = true;
        }
        key += data[i];
        i++;
      }
    }

    key = post_process_key(key);

    i = expect(data, i, ':');
    i = consume_whitespace(data, i);

    // Read unsigned integer value
    auto value_start = i;
    while (i < data.size() && std::isdigit(data[i])) {
      i++;
    }
    auto value = static_cast<int64_t>(
        std::stol(data.substr(value_start, i - value_start)));

    encode_[key] = value;

    i = consume_whitespace(data, i);
    if (i < data.size() && data[i] == ',') {
      i++;
    }

    return i;
  }

  // Assert that the next character in the input string is equal to c. Increment
  // the input string index by one.
  size_t expect(const std::string& data, size_t i, char c) {
    if (i >= data.size() || data[i] != c) {
      std::cerr << "Invalid tokenizer vocabulary file. Expected '" << c
                << "' at index " << i << std::endl;
      exit(1);
    }

    return i + 1;
  }

  std::string post_process_key(std::string key) {
    // Replace the unicode characters with the corresponding byte encoding
    // TODO: adopt byte encoder to handle unicode characters in json file.

    std::unordered_map<std::string, std::string> replacements = {
        {"\\u0120", " "},
        {"\\u010a", "\n"},
    };

    for (const auto& replacement : replacements) {
      size_t pos = 0;
      // While loop through all instances of the substring in the string
      while ((pos = key.find(replacement.first, pos)) != std::string::npos) {
        key.replace(pos, replacement.first.length(), replacement.second);
        pos += replacement.second.length();
      }
    }

    // remove duplicate backslashes
    for (size_t idx = 0; idx < key.length(); idx++) {
      if (key[idx] == '\\') {
        key.erase(idx, 1);
        if (key[idx] == '\\') {
          // If there are two backslashes, keep the second one
          idx += 1;
        }
      }
    }

    return key;
  }
  std::vector<std::string> parse_prompt(const std::string& prompt) {
    std::vector<std::string> result;
    std::string word;
    for (char c : prompt) {
      if (c == ' ') {
        if (!word.empty()) {
          result.push_back(word);
          word.clear();
        }
        word += c;
      } else if (ispunct(c)) {
        if (!word.empty()) {
          result.push_back(word);
          word.clear();
        }
        result.push_back(std::string(1, c));
      } else {
        word += c;
      }
    }
    if (!word.empty()) {
      result.push_back(word);
    }
    return result;
  }
};
