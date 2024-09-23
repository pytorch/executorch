/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>

#include <cstring>

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace executorch {
namespace extension {
namespace llm {

static int compare_tokens(const void* a, const void* b) {
  if (((TokenIndex*)a)->str == nullptr) {
    return -1;
  }
  if (((TokenIndex*)b)->str == nullptr) {
    return 1;
  }
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

BPETokenizer::BPETokenizer() : Tokenizer() {
  for (int i = 0; i < 256; i++) {
    byte_pieces_[i * 2] = (unsigned char)i;
    byte_pieces_[i * 2 + 1] = '\0';
  }
}

/**
 * @brief Load the tokenizer from a file. The tokenizer file contains the
 * vocabulary and scores. The format is: the first integer is the maximum
 * token length, followed by a list of (word_len, word) pairs. Here we
 * are reading all the vocabulary into memory and keep it sorted for fast
 * lookup.
 *
 * @param tokenizer_path The path to the tokenizer file.
 * @return Error
 */
Error BPETokenizer::load(const std::string& tokenizer_path) {
  if (initialized_) {
    ET_LOG(Info, "Tokenizer already initialized");
    return Error::Ok;
  }
  // read in the file
  FILE* file = fopen(tokenizer_path.c_str(), "rb");
  if (!file) {
    ET_LOG(Error, "couldn't load %s", tokenizer_path.c_str());
    return Error::InvalidArgument;
  }
  int32_t metadata[4];
  for (int i = 0; i < 4; i++) {
    if (fread(metadata + i, sizeof(int32_t), 1, file) != 1) {
      ET_LOG(
          Error,
          "Failed to read the metadata at position %d, the tokenizer file is not valid!",
          i);
      return Error::InvalidArgument;
    }
  }

  // now we have two vocab_sizes one from the model and another from the
  // tokenizer file.
  int32_t tokenizer_vocab_size = metadata[0];
  vocab_size_ = tokenizer_vocab_size;
  bos_tok_ = metadata[1];
  eos_tok_ = metadata[2];
  max_token_length_ = metadata[3];

  // allocate space for the vocabulary
  vocab_ = std::make_unique<char*[]>(vocab_size_);
  vocab_scores_ = std::make_unique<float[]>(vocab_size_);
  sorted_vocab_ = std::make_unique<TokenIndex[]>(vocab_size_);

  // read in the vocabulary
  for (int i = 0; i < vocab_size_; i++) {
    if (fread(vocab_scores_.get() + i, sizeof(float), 1, file) != 1) {
      // This is allowed, we just pad the rest of the vocab with <pad> strings
      std::string padding = "<pad>";
      vocab_[i] = new char[padding.length() + 1];
      strcpy(vocab_[i], padding.c_str());
      vocab_[i][padding.length()] = '\0';
      continue;
    }
    int32_t len;
    if (fread(&len, sizeof(int32_t), 1, file) != 1) {
      ET_LOG(Error, "Failed to read the length of the word at index %d", i);
      return Error::InvalidArgument;
    }
    vocab_[i] = new char[len + 1];
    if (fread(vocab_[i], len, 1, file) != 1) {
      ET_LOG(
          Error,
          "Failed to read the word, total length %d, index %d\n",
          len,
          i);
      return Error::InvalidArgument;
    }
    vocab_[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);

  for (int32_t i = 0; i < vocab_size_; i++) {
    sorted_vocab_[i].str = vocab_[i];
    sorted_vocab_[i].id = i;
  }
  qsort(sorted_vocab_.get(), vocab_size_, sizeof(TokenIndex), compare_tokens);

  initialized_ = true;
  return Error::Ok;
}

BPETokenizer::~BPETokenizer() {
  for (int i = 0; i < vocab_size_; i++) {
    delete[] vocab_[i];
  }
}

/**
 * @brief Decode a token into string.
 *
 * @param prev_token The previous token.
 * @param token The current token.
 * @return Result<std::string> A pointer to the string representation of the
 * token.
 */
Result<std::string> BPETokenizer::decode(uint64_t prev_token, uint64_t token)
    const {
  ET_CHECK_OK_OR_RETURN_ERROR(Tokenizer::decode_verify(token));
  const char* piece = vocab_[token];
  // following BOS token, sentencepiece decoder strips any leading
  // whitespace
  if (prev_token == bos_tok_ && piece[0] == ' ') {
    piece++;
  }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char*)byte_pieces_ + byte_val * 2;
  }
  std::string res(piece);
  return res;
}

static int32_t
str_lookup(const char* str, TokenIndex* sorted_vocab, int32_t vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex* res = (TokenIndex*)bsearch(
      &tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != nullptr ? res->id : -1;
}

/**
 * @brief Encode a string into a sequence of tokens.
 *
 * @param text The string to be encoded.
 * @param bos The number of BOS to prepend to the token list.
 * @param eos The number of EOS to append to the token list.
 * @param tokens The output tokens.
 * @param n_tokens The number of tokens.
 * @return Result<std::vector<uint64_t>>
 */
Result<std::vector<uint64_t>>
BPETokenizer::encode(const std::string& text, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    ET_LOG(Error, "Tokenizer not initialized");
    return Error::NotSupported;
  }
  // encode the string text (input) into an upper-bound preallocated tokens[]
  // array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
  // EOS token (=2)
  if (text.empty()) {
    ET_LOG(Error, "cannot encode empty text");
    return Error::InvalidArgument;
  }

  // create a temporary buffer that will store merge candidates of always two
  // consecutive tokens *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  char* str_buffer = new char[max_token_length_ * 2 + 1 + 2];
  size_t str_len = 0;

  // start at 0 tokens
  std::vector<uint64_t> tokens;

  // add optional BOS token, if desired
  if (bos >= 0) {
    while (bos--) {
      tokens.push_back(bos_tok_);
    }
  } else {
    ET_LOG(Error, "bos %d should be >= 0", bos);
    return Error::InvalidArgument;
  }

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have
  // the energy to read more of the sentencepiece code to figure out what it's
  // doing
  const char* space = " ";
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(space, sorted_vocab_.get(), vocab_size_);
    tokens.push_back(dummy_prefix);
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (const char* c = text.c_str(); *c != '\0'; c++) {
    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] =
        *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, sorted_vocab_.get(), vocab_size_);
    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens.push_back(id);
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens.push_back((unsigned char)str_buffer[i] + 3);
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in
  // vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < tokens.size() - 1; i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      snprintf(
          str_buffer,
          max_token_length_ * 2 + 3,
          "%s%s",
          vocab_[tokens[i]],
          vocab_[tokens[i + 1]]);
      int id = str_lookup(str_buffer, sorted_vocab_.get(), vocab_size_);
      if (id != -1 && vocab_scores_[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = vocab_scores_[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx + 1; i < tokens.size() - 1; i++) {
      tokens[i] = tokens[i + 1];
    }
    tokens.pop_back(); // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos >= 0) {
    while (eos--) {
      tokens.push_back(eos_tok_);
    }
  } else {
    ET_LOG(Error, "eos %d should be >= 0", eos);
    return Error::InvalidArgument;
  }

  delete[] str_buffer;
  return Result(tokens);
}

} // namespace llm
} // namespace extension
} // namespace executorch
