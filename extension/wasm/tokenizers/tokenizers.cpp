/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emscripten.h>
#include <emscripten/bind.h>
#include <executorch/runtime/platform/compiler.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>
#include <cstdio>

using namespace emscripten;
using tokenizers::Error;
using tokenizers::HFTokenizer;
using tokenizers::Llama2cTokenizer;
using tokenizers::SPTokenizer;
using tokenizers::Tekken;
using tokenizers::Tiktoken;
using tokenizers::Tokenizer;

#define THROW_JS_ERROR(errorType, message, ...)                           \
  ({                                                                      \
    char msg_buf[256];                                                    \
    int len = snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
    if (len < sizeof(msg_buf)) {                                          \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg_buf);             \
    } else {                                                              \
      std::string msg;                                                    \
      msg.resize(len);                                                    \
      snprintf(&msg[0], len + 1, message, ##__VA_ARGS__);                 \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg.c_str());         \
    }                                                                     \
    __builtin_unreachable();                                              \
  })

/// Throws a JavaScript Error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)          \
  ({                                                 \
    if ET_UNLIKELY ((error) != Error::Ok) {          \
      THROW_JS_ERROR(Error, message, ##__VA_ARGS__); \
    }                                                \
  })

namespace executorch {
namespace extension {
namespace wasm {
namespace tokenizers {

namespace {

#define JS_FORALL_TOKENIZERS(_) \
  _(HFTokenizer)                \
  _(Tiktoken)                   \
  _(SPTokenizer)                \
  _(Llama2cTokenizer)           \
  _(Tekken)

/**
 * EXPERIMENTAL: JavaScript wrapper for Tokenizer.
 */
template <typename T>
class ET_EXPERIMENTAL JsTokenizer {
  static_assert(
      std::is_base_of<Tokenizer, T>::value,
      "T must be a subclass of Tokenizer");

 public:
  JsTokenizer() : tokenizer_(std::make_unique<T>()) {}
  JsTokenizer(const JsTokenizer&) = delete;
  JsTokenizer& operator=(const JsTokenizer&) = delete;
  JsTokenizer(JsTokenizer&&) = default;
  JsTokenizer& operator=(JsTokenizer&&) = default;

  void load_from_uint8_array(val data) {
    // Tokenizer API can't load from a buffer, so we need to write the buffer to
    // a temporary file and load from there.
    static const char* tmpFileName = "tokenizer_input_buffer.tmp";
    FILE* tmp_file = fopen(tmpFileName, "wb");
    if (tmp_file == nullptr) {
      THROW_JS_ERROR(Error, "Failed to open file");
    }
    size_t length = data["length"].as<size_t>();
    std::vector<uint8_t> buffer(length);
    val memory_view = val(typed_memory_view(length, buffer.data()));
    memory_view.call<void>("set", data);
    fwrite(buffer.data(), sizeof(uint8_t), length, tmp_file);
    fclose(tmp_file);
    Error error = tokenizer_->load(tmpFileName);
    THROW_IF_ERROR(error, "Failed to load tokenizer");
    remove(tmpFileName);
  }

  void load(val data) {
    if (data.isString()) {
      Error error = tokenizer_->load(data.as<std::string>());
      THROW_IF_ERROR(error, "Failed to load tokenizer");
    } else if (data.instanceof (val::global("Uint8Array"))) {
      return load_from_uint8_array(data);
    } else if (data.instanceof (val::global("ArrayBuffer"))) {
      return load_from_uint8_array(val::global("Uint8Array").new_(data));
    } else {
      THROW_JS_ERROR(
          TypeError,
          "Unsupported data type: %s",
          data.typeOf().as<std::string>().c_str());
    }
  }

  val encode(const std::string& text, int8_t bos, int8_t eos) const {
    auto res = tokenizer_->encode(text, bos, eos);
    THROW_IF_ERROR(res.error(), "Failed to encode text");
    return val::array(res.get().begin(), res.get().end());
  }

  val encode(const std::string& text, int8_t bos) const {
    return encode(text, bos, 0);
  }

  val encode(const std::string& text) const {
    return encode(text, 0);
  }

  std::string decode(uint64_t prev, uint64_t current) const {
    auto res = tokenizer_->decode(prev, current);
    THROW_IF_ERROR(res.error(), "Failed to decode token");
    return res.get();
  }

  uint64_t vocab_size() const {
    return tokenizer_->vocab_size();
  }

  uint64_t bos_tok() const {
    return tokenizer_->bos_tok();
  }

  uint64_t eos_tok() const {
    return tokenizer_->eos_tok();
  }

  bool is_loaded() const {
    return tokenizer_->is_loaded();
  }

 private:
  std::unique_ptr<T> tokenizer_;
};

} // namespace

EMSCRIPTEN_BINDINGS(TokenizerModule) {
#define JS_BIND_TOKENIZER(NAME)                                           \
  class_<JsTokenizer<NAME>>(#NAME)                                        \
      .constructor<>()                                                    \
      .function("load", &JsTokenizer<NAME>::load)                         \
      .function(                                                          \
          "encode",                                                       \
          select_overload<val(const std::string&) const>(                 \
              &JsTokenizer<NAME>::encode))                                \
      .function(                                                          \
          "encode",                                                       \
          select_overload<val(const std::string&, int8_t) const>(         \
              &JsTokenizer<NAME>::encode))                                \
      .function(                                                          \
          "encode",                                                       \
          select_overload<val(const std::string&, int8_t, int8_t) const>( \
              &JsTokenizer<NAME>::encode))                                \
      .function("decode", &JsTokenizer<NAME>::decode)                     \
      .property("vocabSize", &JsTokenizer<NAME>::vocab_size)              \
      .property("bosTok", &JsTokenizer<NAME>::bos_tok)                    \
      .property("eosTok", &JsTokenizer<NAME>::eos_tok)                    \
      .property("isLoaded", &JsTokenizer<NAME>::is_loaded);
  JS_FORALL_TOKENIZERS(JS_BIND_TOKENIZER)
}

} // namespace tokenizers
} // namespace wasm
} // namespace extension
} // namespace executorch
