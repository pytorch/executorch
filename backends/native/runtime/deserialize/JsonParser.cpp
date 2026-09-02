// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/JsonParser.h>

#include <algorithm>
#include <stdexcept>

namespace ptn {

void JsonValue::throw_bad_kind(const char* expected) {
  throw std::runtime_error(std::string("json: value is not ") + expected);
}

const JsonValue* JsonValue::find(std::string_view key) const {
  const Object& members = as_object();
  const auto it = std::ranges::find(members, key, &JsonMember::key);
  return it == members.end() ? nullptr : &it->value;
}

namespace {

// Guards the recursive descent against a deeply nested document. The documents
// this parses nest two levels; the cap only has to be far above that.
constexpr int kMaxDepth = 64;

constexpr uint32_t kHighSurrogateBegin = 0xd800;
constexpr uint32_t kLowSurrogateBegin = 0xdc00;
constexpr uint32_t kSurrogateEnd = 0xe000;

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

class Parser {
 private:
  std::string_view text_;
  size_t pos_ = 0;

 public:
  explicit Parser(std::string_view text) : text_(text) {}

  JsonValue parse_document() {
    skip_ws();
    JsonValue value = parse_value(0);
    skip_ws();
    if (pos_ != text_.size()) {
      fail("trailing content after the document");
    }
    return value;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(
        "json: " + std::string(what) + " at offset " + std::to_string(pos_));
  }

  bool at_end() const {
    return pos_ >= text_.size();
  }

  char peek() const {
    if (at_end()) {
      fail("unexpected end of document");
    }
    return text_[pos_];
  }

  char take() {
    const char c = peek();
    ++pos_;
    return c;
  }

  void expect(char c, const char* what) {
    if (at_end() || text_[pos_] != c) {
      fail(what);
    }
    ++pos_;
  }

  void skip_ws() {
    while (!at_end()) {
      const char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        break;
      }
      ++pos_;
    }
  }

  void expect_literal(std::string_view literal) {
    if (text_.compare(pos_, literal.size(), literal) != 0) {
      fail("unrecognized literal");
    }
    pos_ += literal.size();
  }

  uint32_t parse_hex4() {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = take();
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        fail("bad hex digit in a \\u escape");
      }
    }
    return value;
  }

  void parse_escape(std::string& out) {
    const char c = take();
    switch (c) {
      case '"':
        out.push_back('"');
        return;
      case '\\':
        out.push_back('\\');
        return;
      case '/':
        out.push_back('/');
        return;
      case 'b':
        out.push_back('\b');
        return;
      case 'f':
        out.push_back('\f');
        return;
      case 'n':
        out.push_back('\n');
        return;
      case 'r':
        out.push_back('\r');
        return;
      case 't':
        out.push_back('\t');
        return;
      case 'u':
        break;
      default:
        fail("unrecognized escape");
    }

    uint32_t cp = parse_hex4();
    if (cp >= kHighSurrogateBegin && cp < kLowSurrogateBegin) {
      // A high surrogate is only meaningful paired with the low one that
      // follows it; anything else would decode to a different character.
      expect('\\', "expected a low surrogate escape after a high surrogate");
      expect('u', "expected a low surrogate escape after a high surrogate");
      const uint32_t low = parse_hex4();
      if (low < kLowSurrogateBegin || low >= kSurrogateEnd) {
        fail("high surrogate not followed by a low surrogate");
      }
      cp = 0x10000 + ((cp - kHighSurrogateBegin) << 10) +
          (low - kLowSurrogateBegin);
    } else if (cp >= kLowSurrogateBegin && cp < kSurrogateEnd) {
      fail("unpaired low surrogate");
    }
    append_utf8(out, cp);
  }

  std::string parse_string() {
    expect('"', "expected a string");
    std::string out;
    while (true) {
      const char c = take();
      if (c == '"') {
        return out;
      }
      if (c == '\\') {
        parse_escape(out);
        continue;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        fail("unescaped control character in a string");
      }
      out.push_back(c);
    }
  }

  JsonValue parse_number() {
    if (peek() == '-') {
      fail("negative numbers are not supported");
    }
    const size_t begin = pos_;
    uint64_t value = 0;
    while (!at_end() && text_[pos_] >= '0' && text_[pos_] <= '9') {
      const uint64_t digit = static_cast<uint64_t>(text_[pos_] - '0');
      if (value > (UINT64_MAX - digit) / 10) {
        fail("number does not fit in 64 bits");
      }
      value = value * 10 + digit;
      ++pos_;
    }
    if (pos_ == begin) {
      fail("expected a number");
    }
    if (!at_end()) {
      const char c = text_[pos_];
      if (c == '.' || c == 'e' || c == 'E') {
        fail("fractional and exponent numbers are not supported");
      }
    }
    return JsonValue(value);
  }

  JsonValue parse_array(int depth) {
    expect('[', "expected an array");
    JsonValue::Array items;
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return JsonValue(std::move(items));
    }
    while (true) {
      skip_ws();
      items.push_back(parse_value(depth + 1));
      skip_ws();
      const char c = take();
      if (c == ']') {
        return JsonValue(std::move(items));
      }
      if (c != ',') {
        fail("expected ',' or ']' in an array");
      }
    }
  }

  JsonValue parse_object(int depth) {
    expect('{', "expected an object");
    JsonValue::Object members;
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return JsonValue(std::move(members));
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':', "expected ':' after an object key");
      skip_ws();
      members.push_back(JsonMember{std::move(key), parse_value(depth + 1)});
      skip_ws();
      const char c = take();
      if (c == '}') {
        return JsonValue(std::move(members));
      }
      if (c != ',') {
        fail("expected ',' or '}' in an object");
      }
    }
  }

  JsonValue parse_value(int depth) {
    if (depth > kMaxDepth) {
      fail("document nests too deeply");
    }
    switch (peek()) {
      case '{':
        return parse_object(depth);
      case '[':
        return parse_array(depth);
      case '"':
        return JsonValue(parse_string());
      case 't':
        expect_literal("true");
        return JsonValue(true);
      case 'f':
        expect_literal("false");
        return JsonValue(false);
      case 'n':
        expect_literal("null");
        return JsonValue();
      default:
        return parse_number();
    }
  }
};

} // namespace

JsonValue json_parse(std::string_view text) {
  Parser parser(text);
  return parser.parse_document();
}

} // namespace ptn
