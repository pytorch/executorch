/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLLMTextRunner.h"

#import "ExecuTorchLLMError.h"

#import <executorch/extension/llm/runner/text_llm_runner.h>
#import <memory>

using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

/// A streaming UTF-8 buffer that accumulates bytes until complete UTF-8
/// sequences are formed. This handles the case where BPE tokenizers output
/// partial multi-byte UTF-8 sequences across token boundaries.
///
/// For example, the Chinese character "清" (UTF-8: E6 B8 85) might be split
/// across two tokens: "æ¸" (E6 B8) and "ħ" (85). This buffer accumulates
/// bytes and only emits complete, valid UTF-8 strings.
class UTF8StreamingBuffer {
public:
  UTF8StreamingBuffer() = default;

  /// Process incoming token bytes and return any complete UTF-8 string.
  /// Returns empty string if more bytes are needed to complete a sequence.
  /// Invalid bytes are silently skipped to maintain robustness.
  std::string process(const std::string& token) {
    buffer_.append(token);

    std::string result;
    size_t i = 0;

    while (i < buffer_.size()) {
      unsigned char byte = static_cast<unsigned char>(buffer_[i]);
      size_t seqLen = utf8SequenceLength(byte);

      if (seqLen == 0) {
        // Invalid start byte (lone continuation or illegal byte) - skip it
        i++;
        continue;
      }

      if (i + seqLen > buffer_.size()) {
        // Incomplete sequence at the end - keep in buffer for next call
        break;
      }

      // Verify all continuation bytes are valid
      bool valid = true;
      for (size_t j = 1; j < seqLen; j++) {
        if (!isUTF8Continuation(static_cast<unsigned char>(buffer_[i + j]))) {
          valid = false;
          break;
        }
      }

      if (valid) {
        // Append complete valid sequence to result
        result.append(buffer_, i, seqLen);
        i += seqLen;
      } else {
        // Invalid sequence - skip only the start byte and resync
        i++;
      }
    }

    // Keep only the incomplete sequence (if any) for next call
    if (i < buffer_.size()) {
      buffer_ = buffer_.substr(i);
    } else {
      buffer_.clear();
    }

    return result;
  }

  /// Flush any remaining bytes in the buffer.
  /// Called at the end of generation to emit any leftover content.
  /// Skips any invalid bytes that couldn't form valid UTF-8.
  std::string flush() {
    std::string result;

    for (size_t i = 0; i < buffer_.size(); i++) {
      unsigned char byte = static_cast<unsigned char>(buffer_[i]);
      size_t seqLen = utf8SequenceLength(byte);

      // Skip invalid start bytes
      if (seqLen == 0) {
        continue;
      }

      // Check if we have enough bytes for this sequence
      if (i + seqLen > buffer_.size()) {
        // Incomplete sequence - skip remaining bytes
        break;
      }

      // Verify continuation bytes
      bool valid = true;
      for (size_t j = 1; j < seqLen; j++) {
        if (!isUTF8Continuation(static_cast<unsigned char>(buffer_[i + j]))) {
          valid = false;
          break;
        }
      }

      if (valid) {
        result.append(buffer_, i, seqLen);
        i += seqLen - 1;  // -1 because loop will i++
      }
    }

    buffer_.clear();
    return result;
  }

private:
  std::string buffer_;

  /// Returns the number of bytes expected for a UTF-8 sequence starting with
  /// the given byte. Returns 0 for invalid start bytes, including overlong
  /// encodings (0xC0, 0xC1) and out-of-range bytes (0xF5-0xFF).
  static size_t utf8SequenceLength(unsigned char byte) {
    if ((byte & 0x80) == 0x00) return 1;        // 0xxxxxxx - ASCII
    if (byte == 0xC0 || byte == 0xC1) return 0; // Overlong encoding - invalid
    if ((byte & 0xE0) == 0xC0) return 2;        // 110xxxxx
    if ((byte & 0xF0) == 0xE0) return 3;        // 1110xxxx
    if (byte >= 0xF5) return 0;                 // Out of Unicode range - invalid
    if ((byte & 0xF8) == 0xF0) return 4;        // 11110xxx
    return 0;  // Continuation byte (10xxxxxx) or other invalid
  }

  /// Returns true if the byte is a valid UTF-8 continuation byte (10xxxxxx).
  static bool isUTF8Continuation(unsigned char byte) {
    return (byte & 0xC0) == 0x80;
  }
};

} // anonymous namespace

@interface ExecuTorchLLMConfig ()

- (const llm::GenerationConfig &)nativeConfig;

@end

@implementation ExecuTorchLLMTextRunner {
  NSString *_modelPath;
  NSString *_tokenizerPath;
  std::unique_ptr<std::vector<std::string>> _specialTokens;
  std::unique_ptr<llm::TextLLMRunner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath {
  return [self initWithModelPath:modelPath
                   tokenizerPath:tokenizerPath
                   specialTokens:@[]];
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath
                    specialTokens:(NSArray<NSString*>*)specialTokens {
  self = [super init];
  if (self) {
    _modelPath = [modelPath copy];
    _tokenizerPath = [tokenizerPath copy];
    _specialTokens = std::make_unique<std::vector<std::string>>();
    for (NSString *token in specialTokens) {
      _specialTokens->emplace_back(token.UTF8String);
    }
  }
  return self;
}

- (BOOL)isLoaded {
  return _runner && _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  if (![self isLoaded]) {
    _runner = llm::create_text_llm_runner(
      _modelPath.UTF8String,
      llm::load_tokenizer(_tokenizerPath.UTF8String, std::move(_specialTokens))
    );
    if (!_runner) {
      if (error) {
        *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                     code:-1
                                 userInfo:@{NSLocalizedDescriptionKey: @"Failed to create runner"}];
      }
      return NO;
    }
  }
  auto status = _runner->load();
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (BOOL)generateWithPrompt:(NSString*)prompt
                    config:(ExecuTorchLLMConfig *)config
             tokenCallback:(nullable void (^)(NSString*))callback
                     error:(NSError**)error {
  if (![self loadWithError:error]) {
    return NO;
  }

  // Create a UTF-8 streaming buffer to handle partial multi-byte sequences.
  // BPE tokenizers (especially ByteLevel like GPT-2/SmolLM) can output tokens
  // that split UTF-8 characters at byte boundaries. This buffer accumulates
  // bytes until complete UTF-8 sequences are formed before calling the callback.
  auto utf8Buffer = std::make_shared<UTF8StreamingBuffer>();

  auto status = _runner->generate(
    prompt.UTF8String,
    config.nativeConfig,
    [callback, utf8Buffer](const std::string& token) {
      if (callback) {
        // Process token through UTF-8 buffer
        std::string validUTF8 = utf8Buffer->process(token);

        // Only call callback when we have complete UTF-8 sequences
        if (!validUTF8.empty()) {
          NSString *tokenString = [[NSString alloc] initWithBytes:validUTF8.data()
                                                           length:validUTF8.size()
                                                         encoding:NSUTF8StringEncoding];
          if (tokenString) {
            callback(tokenString);
          }
        }
      }
    }
  );

  // Flush any remaining bytes in the buffer
  if (callback) {
    std::string remaining = utf8Buffer->flush();
    if (!remaining.empty()) {
      NSString *remainingString = [[NSString alloc] initWithBytes:remaining.data()
                                                           length:remaining.size()
                                                         encoding:NSUTF8StringEncoding];
      if (remainingString) {
        callback(remainingString);
      }
    }
  }

  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (void)stop {
  if (_runner) {
    _runner->stop();
  }
}

- (void)reset {
  if (_runner) {
    _runner->reset();
  }
}

@end
