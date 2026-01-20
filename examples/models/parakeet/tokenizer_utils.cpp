#include "tokenizer_utils.h"

#include <exception>

#include <executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace {

bool is_whitespace_only(const std::string& token) {
  if (token.empty()) {
    return true;
  }

  try {
    const auto codepoints = unicode_cpts_from_utf8(token);
    for (const auto cp : codepoints) {
      if (!unicode_cpt_flags(cp).is_whitespace) {
        return false;
      }
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool is_special_token(const std::string& token) {
  if (token.size() >= 2 && token.front() == '[' && token.back() == ']') {
    return true;
  }
  if (token.size() >= 2 && token.front() == '<' && token.back() == '>') {
    return true;
  }
  if (token.rfind("##", 0) == 0) {
    return true;
  }
  if (token.rfind(u8"▁", 0) == 0) {
    return true;
  }
  if (is_whitespace_only(token)) {
    return true;
  }
  return false;
}

} // namespace

namespace parakeet::tokenizer_utils {

std::unordered_set<std::string> derive_supported_punctuation(
    const tokenizers::Tokenizer& tokenizer) {
  std::unordered_set<std::string> punctuation;

  const int32_t vocab_size = tokenizer.vocab_size();
  for (int32_t id = 0; id < vocab_size; id++) {
    // Use decode to get token text since id_to_piece is not available
    const auto text_result = tokenizer.decode(tokenizer.bos_tok(), static_cast<TokenId>(id));
    if (!text_result.ok()) {
      continue;
    }
    const std::string& piece = text_result.get();
    if (is_special_token(piece)) {
      continue;
    }

    try {
      const auto codepoints = unicode_cpts_from_utf8(piece);
      for (const auto cp : codepoints) {
        if (unicode_cpt_flags(cp).is_punctuation) {
          punctuation.insert(unicode_cpt_to_utf8(cp));
        }
      }
    } catch (const std::exception&) {
      ET_LOG(
          Error,
          "Failed to decode token piece '%s' to codepoints",
          piece.c_str());
    }
  }

  return punctuation;
}

std::string decode_token_sequence(
    const std::vector<TokenId>& tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::string result;
  TokenId prev_token = tokenizer.bos_tok();
  for (const TokenId token : tokens) {
    auto decode_result = tokenizer.decode(prev_token, token);
    if (decode_result.ok()) {
      result += decode_result.get();
    }
    prev_token = token;
  }
  return result;
}

std::string decode_token_sequence(
    const std::vector<Token>& decoded_tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::vector<TokenId> token_ids;
  token_ids.reserve(decoded_tokens.size());
  for (const auto& tok : decoded_tokens) {
    token_ids.push_back(tok.id);
  }
  return decode_token_sequence(token_ids, tokenizer);
}

} // namespace parakeet::tokenizer_utils
