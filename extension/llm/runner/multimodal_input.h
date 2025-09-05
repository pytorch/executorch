/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
// A generic multimodal input class that can hold either image or text data.

#pragma once

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/runtime/platform/compiler.h>
#include <string>
#include <variant>

namespace executorch::extension::llm {

/**
 * A generic class to hold either image, text, or audio data for multimodal
 * inputs. This allows the generate() API to take a std::vector of these objects
 * instead of separate image, text, and audio parameters.
 */
class ET_EXPERIMENTAL MultimodalInput {
 public:
  /// Type of multimodal input data
  enum class Type {
    TEXT, ///< Text string input
    IMAGE, ///< Processed image input
    AUDIO, ///< Processed audio input
    RAW_AUDIO, ///< Raw unprocessed audio input (straight from audio file)
    UNSUPPORTED ///< Unsupported input type
  };

  // Constructors
  explicit MultimodalInput(const std::string& text) : data_(text) {}
  explicit MultimodalInput(std::string&& text) : data_(std::move(text)) {}
  explicit MultimodalInput(const Image& image) : data_(image) {}
  explicit MultimodalInput(Image&& image) : data_(std::move(image)) {}
  explicit MultimodalInput(const Audio& audio) : data_(audio) {}
  explicit MultimodalInput(Audio&& audio) : data_(std::move(audio)) {}
  explicit MultimodalInput(const RawAudio& raw_audio) : data_(raw_audio) {}
  explicit MultimodalInput(RawAudio&& raw_audio)
      : data_(std::move(raw_audio)) {}

  // Copy constructor and assignment
  MultimodalInput(const MultimodalInput& other) = default;
  MultimodalInput& operator=(const MultimodalInput& other) = default;

  // Move constructor and assignment
  MultimodalInput(MultimodalInput&& other) noexcept = default;
  MultimodalInput& operator=(MultimodalInput&& other) noexcept = default;

  // Destructor
  ~MultimodalInput() = default;

  /**
   * Check if this input contains text data.
   * @return true if this input contains text, false otherwise.
   */
  bool is_text() const noexcept {
    return std::holds_alternative<std::string>(data_);
  }

  /**
   * Check if this input contains image data.
   * @return true if this input contains an image, false otherwise.
   */
  bool is_image() const noexcept {
    return std::holds_alternative<Image>(data_);
  }

  /**
   * Check if this input contains audio data.
   * @return true if this input contains audio, false otherwise.
   */
  bool is_audio() const noexcept {
    return std::holds_alternative<Audio>(data_);
  }

  /**
   * Check if this input contains raw audio data.
   * @return true if this input contains raw audio, false otherwise.
   */
  bool is_raw_audio() const noexcept {
    return std::holds_alternative<RawAudio>(data_);
  }

  /**
   * Get the type of data stored in this input.
   * @return Type::TEXT if text data, Type::IMAGE if image data, Type::AUDIO if
   * audio data, Type::RAW_AUDIO if raw audio data.
   */
  Type get_type() const noexcept {
    if (is_text())
      return Type::TEXT;
    if (is_image())
      return Type::IMAGE;
    if (is_audio())
      return Type::AUDIO;
    if (is_raw_audio())
      return Type::RAW_AUDIO;
    return Type::UNSUPPORTED;
  }

  /**
   * Get the text data from this input.
   * @return Reference to the stored text string.
   * @throws std::bad_variant_access if this input doesn't contain text.
   */
  const std::string& get_text() const& {
    return std::get<std::string>(data_);
  }

  /**
   * Get the text data from this input (mutable version).
   * @return Mutable reference to the stored text string.
   * @throws std::bad_variant_access if this input doesn't contain text.
   */
  std::string& get_text() & {
    return std::get<std::string>(data_);
  }

  /**
   * Get the text data from this input (rvalue version).
   * @return Rvalue reference to the stored text string for efficient moves.
   * @throws std::bad_variant_access if this input doesn't contain text.
   */
  std::string&& get_text() && {
    return std::get<std::string>(std::move(data_));
  }

  /**
   * Get the image data from this input.
   * @return Reference to the stored Image object.
   * @throws std::bad_variant_access if this input doesn't contain an image.
   */
  const Image& get_image() const& {
    return std::get<Image>(data_);
  }

  /**
   * Get the image data from this input (mutable version).
   * @return Mutable reference to the stored Image object.
   * @throws std::bad_variant_access if this input doesn't contain an image.
   */
  Image& get_image() & {
    return std::get<Image>(data_);
  }

  /**
   * Get the image data from this input (rvalue version).
   * @return Rvalue reference to the stored Image object for efficient moves.
   * @throws std::bad_variant_access if this input doesn't contain an image.
   */
  Image&& get_image() && {
    return std::get<Image>(std::move(data_));
  }

  /**
   * Get the audio data from this input.
   * @return Reference to the stored Audio object.
   * @throws std::bad_variant_access if this input doesn't contain audio.
   */
  const Audio& get_audio() const& {
    return std::get<Audio>(data_);
  }

  /**
   * Get the audio data from this input (mutable version).
   * @return Mutable reference to the stored Audio object.
   * @throws std::bad_variant_access if this input doesn't contain audio.
   */
  Audio& get_audio() & {
    return std::get<Audio>(data_);
  }

  /**
   * Get the audio data from this input (rvalue version).
   * @return Rvalue reference to the stored Audio object for efficient moves.
   * @throws std::bad_variant_access if this input doesn't contain audio.
   */
  Audio&& get_audio() && {
    return std::get<Audio>(std::move(data_));
  }

  /**
   * Get the raw audio data from this input.
   * @return Reference to the stored RawAudio object.
   * @throws std::bad_variant_access if this input doesn't contain raw audio.
   */
  const RawAudio& get_raw_audio() const& {
    return std::get<RawAudio>(data_);
  }

  /**
   * Get the raw audio data from this input (mutable version).
   * @return Mutable reference to the stored RawAudio object.
   * @throws std::bad_variant_access if this input doesn't contain raw audio.
   */
  RawAudio& get_raw_audio() & {
    return std::get<RawAudio>(data_);
  }

  /**
   * Get the raw audio data from this input (rvalue version).
   * @return Rvalue reference to the stored RawAudio object for efficient moves.
   * @throws std::bad_variant_access if this input doesn't contain raw audio.
   */
  RawAudio&& get_raw_audio() && {
    return std::get<RawAudio>(std::move(data_));
  }

  /**
   * Try to get the text data from this input safely.
   * @return Pointer to the text string if this input contains text, nullptr
   * otherwise.
   */
  const std::string* try_get_text() const noexcept {
    return std::get_if<std::string>(&data_);
  }

  /**
   * Try to get the text data from this input safely (mutable version).
   * @return Pointer to the text string if this input contains text, nullptr
   * otherwise.
   */
  std::string* try_get_text() noexcept {
    return std::get_if<std::string>(&data_);
  }

  /**
   * Try to get the image data from this input safely.
   * @return Pointer to the Image object if this input contains an image,
   * nullptr otherwise.
   */
  const Image* try_get_image() const noexcept {
    return std::get_if<Image>(&data_);
  }

  /**
   * Try to get the image data from this input safely (mutable version).
   * @return Pointer to the Image object if this input contains an image,
   * nullptr otherwise.
   */
  Image* try_get_image() noexcept {
    return std::get_if<Image>(&data_);
  }

  /**
   * Try to get the audio data from this input safely.
   * @return Pointer to the Audio object if this input contains audio,
   * nullptr otherwise.
   */
  const Audio* try_get_audio() const noexcept {
    return std::get_if<Audio>(&data_);
  }

  /**
   * Try to get the audio data from this input safely (mutable version).
   * @return Pointer to the Audio object if this input contains audio,
   * nullptr otherwise.
   */
  Audio* try_get_audio() noexcept {
    return std::get_if<Audio>(&data_);
  }

  /**
   * Try to get the raw audio data from this input safely.
   * @return Pointer to the RawAudio object if this input contains raw audio,
   * nullptr otherwise.
   */
  const RawAudio* try_get_raw_audio() const noexcept {
    return std::get_if<RawAudio>(&data_);
  }

  /**
   * Try to get the raw audio data from this input safely (mutable version).
   * @return Pointer to the RawAudio object if this input contains raw audio,
   * nullptr otherwise.
   */
  RawAudio* try_get_raw_audio() noexcept {
    return std::get_if<RawAudio>(&data_);
  }

 private:
  std::variant<std::string, Image, Audio, RawAudio> data_;
};

// Convenience factory functions
inline MultimodalInput make_text_input(const std::string& text) noexcept {
  return MultimodalInput(text);
}

inline MultimodalInput make_text_input(std::string&& text) noexcept {
  return MultimodalInput(std::move(text));
}

inline MultimodalInput make_image_input(const Image& image) noexcept {
  return MultimodalInput(image);
}

inline MultimodalInput make_image_input(Image&& image) noexcept {
  return MultimodalInput(std::move(image));
}

inline MultimodalInput make_audio_input(const Audio& audio) noexcept {
  return MultimodalInput(audio);
}

inline MultimodalInput make_audio_input(Audio&& audio) noexcept {
  return MultimodalInput(std::move(audio));
}

inline MultimodalInput make_raw_audio_input(
    const RawAudio& raw_audio) noexcept {
  return MultimodalInput(raw_audio);
}

inline MultimodalInput make_raw_audio_input(RawAudio&& raw_audio) noexcept {
  return MultimodalInput(std::move(raw_audio));
}

} // namespace executorch::extension::llm
