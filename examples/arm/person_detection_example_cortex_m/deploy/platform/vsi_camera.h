/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

constexpr uint32_t kCameraWidth = 320;
constexpr uint32_t kCameraHeight = 240;
constexpr uint32_t kCameraChannels = 3;
constexpr uint32_t kCameraFrameSize =
    kCameraWidth * kCameraHeight * kCameraChannels;

enum class CameraCaptureResult {
  Frame,
  EndOfStream,
  Error,
};

bool camera_init(uint8_t* frame_buffer, uint32_t frame_buffer_size);
CameraCaptureResult camera_capture();
