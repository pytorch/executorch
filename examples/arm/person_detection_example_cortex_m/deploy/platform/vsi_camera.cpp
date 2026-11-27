/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vsi_camera.h"

#include "ARMCM55.h"

#include <cstddef>

namespace {

struct VsiRegisters {
  struct {
    volatile uint32_t enable;
    volatile uint32_t set;
    volatile uint32_t clear;
    volatile const uint32_t status;
  } irq;
  uint32_t reserved1[60];
  struct {
    volatile uint32_t control;
    volatile uint32_t interval;
    volatile const uint32_t count;
  } timer;
  uint32_t reserved2[61];
  struct {
    volatile uint32_t control;
    volatile uint32_t address;
    volatile uint32_t block_size;
    volatile uint32_t block_count;
    volatile const uint32_t block_index;
  } dma;
  uint32_t reserved3[59];
  volatile uint32_t user[64];
};

constexpr uintptr_t kVsiCameraBase = 0x4FF40000;
auto* const camera = reinterpret_cast<VsiRegisters*>(kVsiCameraBase);

constexpr uint32_t kModeInput = 0;
constexpr uint32_t kControlEnable = 1U << 0;
constexpr uint32_t kStatusActive = 1U << 0;
constexpr uint32_t kStatusBufferEmpty = 1U << 1;
constexpr uint32_t kStatusOverflow = 1U << 3;
constexpr uint32_t kStatusUnderflow = 1U << 4;
constexpr uint32_t kStatusEndOfStream = 1U << 5;
constexpr uint32_t kDmaEnable = 1U << 0;
constexpr uint32_t kTimerRun = 1U << 0;
constexpr uint32_t kTimerTriggerDma = 1U << 3;
constexpr uint32_t kRgb888 = 2;

uint8_t* frame = nullptr;

} // namespace

bool camera_init(uint8_t* frame_buffer, uint32_t frame_buffer_size) {
  if (frame_buffer == nullptr || frame_buffer_size != kCameraFrameSize) {
    return false;
  }
  frame = frame_buffer;
  camera->timer.control = 0;
  camera->dma.control = 0;
  camera->irq.clear = 0xFFFFFFFF;
  camera->irq.enable = 0;
  camera->user[0] = kModeInput;
  camera->user[1] = 0;
  camera->user[6] = kCameraWidth;
  camera->user[7] = kCameraHeight;
  camera->user[8] = kRgb888;
  camera->user[9] = 10;
  camera->user[12] = 1;
  camera->timer.interval = 100000;
  camera->dma.address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(frame));
  camera->dma.block_size = kCameraFrameSize;
  camera->dma.block_count = 1;
  __DSB();
  return true;
}

CameraCaptureResult camera_capture() {
  if (frame == nullptr) {
    return CameraCaptureResult::Error;
  }

  camera->timer.control = 0;
  camera->dma.control = 0;
  camera->user[1] = kControlEnable;
  camera->dma.control = kDmaEnable;
  camera->timer.control = kTimerRun | kTimerTriggerDma;
  __DSB();

  for (;;) {
    const uint32_t status = camera->user[2];
    if ((status & kStatusEndOfStream) != 0) {
      camera->timer.control = 0;
      camera->dma.control = 0;
      return CameraCaptureResult::EndOfStream;
    }
    if ((status & (kStatusOverflow | kStatusUnderflow)) != 0) {
      camera->timer.control = 0;
      camera->dma.control = 0;
      return CameraCaptureResult::Error;
    }
    if ((status & kStatusBufferEmpty) == 0 && (status & kStatusActive) == 0) {
      camera->timer.control = 0;
      camera->dma.control = 0;
      camera->user[1] = 0;
      __DSB();
      return CameraCaptureResult::Frame;
    }
  }
}
