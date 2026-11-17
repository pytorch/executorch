/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <camera_stream.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <driver/usb_serial_jtag.h>
#include <driver/usb_serial_jtag_vfs.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_rom_crc.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <img_converters.h>
#include <jpeg_decoder.h>
#include <sdkconfig.h>

#include <et_model.h>

extern "C" {
extern const uint8_t camera_model_start[];
extern const uint8_t camera_model_end[];
}

namespace executorch {
namespace examples {
namespace espressif {

namespace {

constexpr int TOP_K = 5;
constexpr int CROP_NUMERATOR = 224;
constexpr int CROP_DENOMINATOR = 256;
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
constexpr size_t USB_TX_BUFFER_BYTES = 8 * 1024;
constexpr size_t SENSOR_JPEG_BUFFER_BYTES = 32 * 1024;
constexpr TickType_t USB_WRITE_TIMEOUT = pdMS_TO_TICKS(200);
constexpr size_t RESULT_QUEUE_LENGTH = 4;

#if CONFIG_CAMERA_STREAM_VFLIP
constexpr int VFLIP = 1;
#else
constexpr int VFLIP = 0;
#endif
#if CONFIG_CAMERA_STREAM_HMIRROR
constexpr int HMIRROR = 1;
#else
constexpr int HMIRROR = 0;
#endif
#if CONFIG_CAMERA_STREAM_PREVIEW
constexpr bool PREVIEW = true;
constexpr int PREVIEW_QUALITY = CONFIG_CAMERA_STREAM_PREVIEW_QUALITY;
constexpr int PREVIEW_MAX_FPS = CONFIG_CAMERA_STREAM_PREVIEW_MAX_FPS;
#else
constexpr bool PREVIEW = false;
constexpr int PREVIEW_QUALITY = 50;
constexpr int PREVIEW_MAX_FPS = 1;
#endif
#if CONFIG_ESPNN_DELEGATE_DUAL_CORE
constexpr bool DUAL_CORE = true;
#else
constexpr bool DUAL_CORE = false;
#endif
#if CONFIG_CAMERA_STREAM_SENSOR_JPEG
constexpr bool SENSOR_JPEG = true;
constexpr int SENSOR_JPEG_QUALITY = CONFIG_CAMERA_STREAM_SENSOR_JPEG_QUALITY;
#else
constexpr bool SENSOR_JPEG = false;
constexpr int SENSOR_JPEG_QUALITY = 12;
#endif

struct Pinout {
  const char* name;
  int pwdn, reset, xclk, sda, scl;
  int d7, d6, d5, d4, d3, d2, d1, d0;
  int vsync, href, pclk;
};

#if CONFIG_CAMERA_STREAM_PINOUT_XIAO_SENSE
constexpr Pinout PINOUT = {
    "XIAO ESP32S3 Sense",
    -1,
    -1,
    10,
    40,
    39,
    48,
    11,
    12,
    14,
    16,
    18,
    17,
    15,
    38,
    47,
    13};
#else
constexpr Pinout PINOUT = {
    "ESP32-S3-EYE/Freenove",
    -1,
    -1,
    15,
    4,
    5,
    16,
    17,
    18,
    12,
    10,
    8,
    9,
    11,
    6,
    7,
    13};
#endif

struct Prediction {
  int32_t index;
  float probability;
};

struct Result {
  uint32_t seq;
  int64_t decode_us;
  int64_t preprocess_us;
  int64_t invoke_us;
  int64_t delegate_us;
  int64_t cycle_us;
  int64_t latency_us;
  uint32_t dropped_results;
  uint32_t decode_errors;
  Prediction top[TOP_K];
};

struct Shared {
  SemaphoreHandle_t state_lock;
  QueueHandle_t results;
  EtModel* model;
  int32_t input_width;
  int32_t input_height;
  bool decoder_swaps_red_blue;
  uint8_t* published_bgr;
  bool published_ready;
  uint32_t published_seq;
};

Shared g_shared{};

void* psram_alloc(size_t bytes) {
  void* pointer = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (pointer == nullptr)
    pointer = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
  return pointer;
}

int64_t frame_time_us(const camera_fb_t* frame) {
  return static_cast<int64_t>(frame->timestamp.tv_sec) * 1000000LL +
      frame->timestamp.tv_usec;
}

bool decode_jpeg_rgb888(
    const uint8_t* jpeg,
    size_t length,
    uint8_t* output,
    size_t output_size,
    int32_t& width,
    int32_t& height) {
  esp_jpeg_image_cfg_t config{};
  config.indata = const_cast<uint8_t*>(jpeg);
  config.indata_size = static_cast<uint32_t>(length);
  config.outbuf = output;
  config.outbuf_size = static_cast<uint32_t>(output_size);
  config.out_format = JPEG_IMAGE_FORMAT_RGB888;
  config.out_scale = JPEG_IMAGE_SCALE_0;
  esp_jpeg_image_output_t image{};
  if (esp_jpeg_decode(&config, &image) != ESP_OK)
    return false;
  width = image.width;
  height = image.height;
  return true;
}

bool probe_decoder_channel_order(bool& swaps_red_blue) {
  constexpr int SIDE = 16;
  uint8_t bgr[SIDE * SIDE * 3];
  for (int i = 0; i < SIDE * SIDE; ++i) {
    bgr[i * 3 + 0] = 0;
    bgr[i * 3 + 1] = 0;
    bgr[i * 3 + 2] = 255;
  }
  uint8_t* jpeg = nullptr;
  size_t length = 0;
  if (!fmt2jpg(
          bgr, sizeof(bgr), SIDE, SIDE, PIXFORMAT_RGB888, 90, &jpeg, &length))
    return false;
  uint8_t decoded[SIDE * SIDE * 3];
  int32_t width = 0;
  int32_t height = 0;
  const bool ok =
      decode_jpeg_rgb888(jpeg, length, decoded, sizeof(decoded), width, height);
  std::free(jpeg);
  if (!ok || width != SIDE || height != SIDE)
    return false;
  const int center = (SIDE / 2 * SIDE + SIDE / 2) * 3;
  std::printf(
      "Decoder check: pure red decodes to bytes %u,%u,%u\n",
      decoded[center],
      decoded[center + 1],
      decoded[center + 2]);
  if (decoded[center] > 200 && decoded[center + 2] < 60) {
    swaps_red_blue = false;
    return true;
  }
  if (decoded[center + 2] > 200 && decoded[center] < 60) {
    swaps_red_blue = true;
    return true;
  }
  return false;
}

constexpr char BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t base64_encode(const uint8_t* source, size_t length, char* destination) {
  size_t o = 0;
  size_t i = 0;
  for (; i + 2 < length; i += 3) {
    const uint32_t v = (static_cast<uint32_t>(source[i]) << 16) |
        (static_cast<uint32_t>(source[i + 1]) << 8) | source[i + 2];
    destination[o++] = BASE64_ALPHABET[(v >> 18) & 63];
    destination[o++] = BASE64_ALPHABET[(v >> 12) & 63];
    destination[o++] = BASE64_ALPHABET[(v >> 6) & 63];
    destination[o++] = BASE64_ALPHABET[v & 63];
  }
  if (i < length) {
    uint32_t v = static_cast<uint32_t>(source[i]) << 16;
    if (i + 1 < length)
      v |= static_cast<uint32_t>(source[i + 1]) << 8;
    destination[o++] = BASE64_ALPHABET[(v >> 18) & 63];
    destination[o++] = BASE64_ALPHABET[(v >> 12) & 63];
    destination[o++] = i + 1 < length ? BASE64_ALPHABET[(v >> 6) & 63] : '=';
    destination[o++] = '=';
  }
  return o;
}

class UsbLink {
 public:
  static bool begin() {
    usb_serial_jtag_driver_config_t config =
        USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    config.tx_buffer_size = USB_TX_BUFFER_BYTES;
    config.rx_buffer_size = 256;
    if (!usb_serial_jtag_is_driver_installed() &&
        usb_serial_jtag_driver_install(&config) != ESP_OK)
      return false;
    std::fflush(stdout);
    usb_serial_jtag_vfs_use_driver();
    return true;
  }

  bool write_line(const char* data, size_t length) {
    size_t sent = 0;
    while (sent < length) {
      const int written = usb_serial_jtag_write_bytes(
          data + sent, length - sent, USB_WRITE_TIMEOUT);
      if (written <= 0) {
        if (sent != 0)
          usb_serial_jtag_write_bytes("\n", 1, 0);
        host_reading_ = false;
        ++failed_writes_;
        return false;
      }
      sent += static_cast<size_t>(written);
    }
    host_reading_ = true;
    return true;
  }

  bool probe() {
    if (usb_serial_jtag_write_bytes("\n", 1, USB_WRITE_TIMEOUT) > 0)
      host_reading_ = true;
    return host_reading_;
  }

  bool host_reading() const {
    return host_reading_;
  }
  uint32_t failed_writes() const {
    return failed_writes_;
  }

 private:
  bool host_reading_ = true;
  uint32_t failed_writes_ = 0;
};

class JpegLine {
 public:
  bool reserve(size_t bytes) {
    if (bytes <= capacity_)
      return true;
    std::free(buffer_);
    buffer_ = static_cast<char*>(psram_alloc(bytes));
    capacity_ = buffer_ != nullptr ? bytes : 0;
    return buffer_ != nullptr;
  }

  size_t build(
      const char* kind,
      uint32_t seq,
      uint32_t result_seq,
      int32_t width,
      int32_t height,
      int64_t encode_us,
      const uint8_t* jpeg,
      size_t length) {
    const size_t needed = 192 + ((length + 2) / 3) * 4;
    if (!reserve(needed))
      return 0;
    const uint32_t crc = esp_rom_crc32_le(0, jpeg, length);
    const int header = std::snprintf(
        buffer_,
        capacity_,
        "#JPG kind=%s seq=%u res=%u w=%ld h=%ld enc_us=%lld len=%u crc=%08lx b64=",
        kind,
        static_cast<unsigned>(seq),
        static_cast<unsigned>(result_seq),
        static_cast<long>(width),
        static_cast<long>(height),
        static_cast<long long>(encode_us),
        static_cast<unsigned>(length),
        static_cast<unsigned long>(crc));
    if (header <= 0)
      return 0;
    size_t total = static_cast<size_t>(header);
    total += base64_encode(jpeg, length, buffer_ + total);
    buffer_[total++] = '\n';
    return total;
  }

  bool send(UsbLink& link, size_t length) const {
    return length != 0 && link.write_line(buffer_, length);
  }

 private:
  char* buffer_ = nullptr;
  size_t capacity_ = 0;
};

uint8_t expand5(uint32_t v) {
  return static_cast<uint8_t>((v << 3) | (v >> 2));
}
uint8_t expand6(uint32_t v) {
  return static_cast<uint8_t>((v << 2) | (v >> 4));
}

class Resampler {
 public:
  bool init(
      int32_t source_width,
      int32_t source_height,
      int32_t output_width,
      int32_t output_height) {
    source_width_ = source_width;
    output_width_ = output_width;
    output_height_ = output_height;
    int32_t side = source_width < source_height ? source_width : source_height;
    side = side * CROP_NUMERATOR / CROP_DENOMINATOR;
    crop_x_ = (source_width - side) / 2;
    crop_y_ = (source_height - side) / 2;
    crop_side_ = side;
    x_index_ =
        static_cast<int32_t*>(std::malloc(sizeof(int32_t) * output_width * 2));
    y_index_ =
        static_cast<int32_t*>(std::malloc(sizeof(int32_t) * output_height * 2));
    x_weight_ =
        static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * output_width));
    y_weight_ =
        static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * output_height));
    if (x_index_ == nullptr || y_index_ == nullptr || x_weight_ == nullptr ||
        y_weight_ == nullptr)
      return false;
    build(output_width, crop_x_, x_index_, x_weight_);
    build(output_height, crop_y_, y_index_, y_weight_);
    return true;
  }

  void run_rgb565_be(const uint8_t* source, uint8_t* rgb888) const {
    for (int32_t oy = 0; oy < output_height_; ++oy) {
      const uint8_t* row0 =
          source + static_cast<size_t>(y_index_[oy * 2]) * source_width_ * 2;
      const uint8_t* row1 = source +
          static_cast<size_t>(y_index_[oy * 2 + 1]) * source_width_ * 2;
      const uint32_t wy = y_weight_[oy];
      uint8_t* out = rgb888 + static_cast<size_t>(oy) * output_width_ * 3;
      for (int32_t ox = 0; ox < output_width_; ++ox) {
        const int32_t x0 = x_index_[ox * 2] * 2;
        const int32_t x1 = x_index_[ox * 2 + 1] * 2;
        const uint32_t wx = x_weight_[ox];
        const uint32_t p00 =
            (static_cast<uint32_t>(row0[x0]) << 8) | row0[x0 + 1];
        const uint32_t p01 =
            (static_cast<uint32_t>(row0[x1]) << 8) | row0[x1 + 1];
        const uint32_t p10 =
            (static_cast<uint32_t>(row1[x0]) << 8) | row1[x0 + 1];
        const uint32_t p11 =
            (static_cast<uint32_t>(row1[x1]) << 8) | row1[x1 + 1];
        out[ox * 3 + 0] = blend(
            expand5(p00 >> 11),
            expand5(p01 >> 11),
            expand5(p10 >> 11),
            expand5(p11 >> 11),
            wx,
            wy);
        out[ox * 3 + 1] = blend(
            expand6((p00 >> 5) & 63),
            expand6((p01 >> 5) & 63),
            expand6((p10 >> 5) & 63),
            expand6((p11 >> 5) & 63),
            wx,
            wy);
        out[ox * 3 + 2] = blend(
            expand5(p00 & 31),
            expand5(p01 & 31),
            expand5(p10 & 31),
            expand5(p11 & 31),
            wx,
            wy);
      }
    }
  }

  void run_rgb888(const uint8_t* source, bool swap_red_blue, uint8_t* rgb888)
      const {
    const int red = swap_red_blue ? 2 : 0;
    const int blue = swap_red_blue ? 0 : 2;
    for (int32_t oy = 0; oy < output_height_; ++oy) {
      const uint8_t* row0 =
          source + static_cast<size_t>(y_index_[oy * 2]) * source_width_ * 3;
      const uint8_t* row1 = source +
          static_cast<size_t>(y_index_[oy * 2 + 1]) * source_width_ * 3;
      const uint32_t wy = y_weight_[oy];
      uint8_t* out = rgb888 + static_cast<size_t>(oy) * output_width_ * 3;
      for (int32_t ox = 0; ox < output_width_; ++ox) {
        const int32_t x0 = x_index_[ox * 2] * 3;
        const int32_t x1 = x_index_[ox * 2 + 1] * 3;
        const uint32_t wx = x_weight_[ox];
        out[ox * 3 + 0] = blend(
            row0[x0 + red],
            row0[x1 + red],
            row1[x0 + red],
            row1[x1 + red],
            wx,
            wy);
        out[ox * 3 + 1] = blend(
            row0[x0 + 1], row0[x1 + 1], row1[x0 + 1], row1[x1 + 1], wx, wy);
        out[ox * 3 + 2] = blend(
            row0[x0 + blue],
            row0[x1 + blue],
            row1[x0 + blue],
            row1[x1 + blue],
            wx,
            wy);
      }
    }
  }

  int32_t crop_x() const {
    return crop_x_;
  }
  int32_t crop_y() const {
    return crop_y_;
  }
  int32_t crop_side() const {
    return crop_side_;
  }

 private:
  static uint8_t blend(
      uint32_t a,
      uint32_t b,
      uint32_t c,
      uint32_t d,
      uint32_t wx,
      uint32_t wy) {
    const uint32_t top = a * (256 - wx) + b * wx;
    const uint32_t bottom = c * (256 - wx) + d * wx;
    return static_cast<uint8_t>((top * (256 - wy) + bottom * wy + 32768) >> 16);
  }

  void build(int32_t output, int32_t offset, int32_t* index, uint16_t* weight)
      const {
    const int32_t last = crop_side_ - 1;
    for (int32_t o = 0; o < output; ++o) {
      int32_t position =
          static_cast<int32_t>(
              (static_cast<int64_t>(2 * o + 1) * crop_side_ * 256) /
              (2 * output)) -
          128;
      if (position < 0)
        position = 0;
      int32_t i0 = position >> 8;
      if (i0 > last)
        i0 = last;
      const int32_t i1 = i0 < last ? i0 + 1 : last;
      index[o * 2] = offset + i0;
      index[o * 2 + 1] = offset + i1;
      weight[o] = static_cast<uint16_t>(i0 < last ? position & 255 : 0);
    }
  }

  int32_t source_width_ = 0;
  int32_t output_width_ = 0;
  int32_t output_height_ = 0;
  int32_t crop_x_ = 0;
  int32_t crop_y_ = 0;
  int32_t crop_side_ = 0;
  int32_t* x_index_ = nullptr;
  int32_t* y_index_ = nullptr;
  uint16_t* x_weight_ = nullptr;
  uint16_t* y_weight_ = nullptr;
};

bool top_k(const float* logits, float* scratch, size_t count, Prediction* top) {
  if (logits == nullptr)
    return false;
  std::memcpy(scratch, logits, sizeof(float) * count);
  float maximum = scratch[0];
  for (size_t i = 1; i < count; ++i) {
    if (scratch[i] > maximum)
      maximum = scratch[i];
  }
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    scratch[i] = std::exp(scratch[i] - maximum);
    sum += scratch[i];
  }
  for (int k = 0; k < TOP_K; ++k) {
    top[k].index = -1;
    top[k].probability = -1.0f;
  }
  for (size_t i = 0; i < count; ++i) {
    const float p = static_cast<float>(scratch[i] / sum);
    if (p <= top[TOP_K - 1].probability)
      continue;
    int k = TOP_K - 1;
    while (k > 0 && p > top[k - 1].probability) {
      top[k] = top[k - 1];
      --k;
    }
    top[k].index = static_cast<int32_t>(i);
    top[k].probability = p;
  }
  return true;
}

void inference_task(void*) {
  EtModel& model = *g_shared.model;
  const size_t classes = model.info().classes;
  const int32_t width = g_shared.input_width;
  const int32_t height = g_shared.input_height;
  const size_t rgb_bytes = static_cast<size_t>(width) * height * 3;
  uint8_t* rgb = static_cast<uint8_t*>(psram_alloc(rgb_bytes));
  float* scratch = static_cast<float*>(std::malloc(sizeof(float) * classes));
  uint8_t* decoded = nullptr;
  size_t decoded_size = 0;
  Resampler resampler;
  bool resampler_ready = false;
  if (rgb == nullptr || scratch == nullptr) {
    std::printf("#ERR inference buffers could not be allocated\n");
    vTaskDelete(nullptr);
    return;
  }
  uint32_t seq = 0;
  uint32_t dropped = 0;
  uint32_t decode_errors = 0;
  int64_t previous = esp_timer_get_time();
  while (true) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr)
      continue;
    const int64_t started = esp_timer_get_time();
    const int64_t captured = frame_time_us(frame);
    const int32_t frame_width = static_cast<int32_t>(frame->width);
    const int32_t frame_height = static_cast<int32_t>(frame->height);
    if (!resampler_ready) {
      resampler_ready =
          resampler.init(frame_width, frame_height, width, height);
      std::printf(
          "#CROP x=%ld y=%ld side=%ld\n",
          static_cast<long>(resampler.crop_x()),
          static_cast<long>(resampler.crop_y()),
          static_cast<long>(resampler.crop_side()));
    }
    int64_t decode_us = 0;
    if (SENSOR_JPEG) {
      const size_t needed = static_cast<size_t>(frame_width) * frame_height * 3;
      if (decoded_size < needed) {
        std::free(decoded);
        decoded = static_cast<uint8_t*>(psram_alloc(needed));
        decoded_size = decoded != nullptr ? needed : 0;
      }
      int32_t decoded_width = 0;
      int32_t decoded_height = 0;
      const bool ok = decoded != nullptr &&
          decode_jpeg_rgb888(
                          frame->buf,
                          frame->len,
                          decoded,
                          decoded_size,
                          decoded_width,
                          decoded_height) &&
          decoded_width == frame_width && decoded_height == frame_height;
      esp_camera_fb_return(frame);
      decode_us = esp_timer_get_time() - started;
      if (!ok) {
        ++decode_errors;
        continue;
      }
      resampler.run_rgb888(decoded, g_shared.decoder_swaps_red_blue, rgb);
    } else {
      resampler.run_rgb565_be(frame->buf, rgb);
      esp_camera_fb_return(frame);
    }
    model.set_input_rgb888(rgb, IMAGENET_MEAN, IMAGENET_STD);
    const int64_t preprocessed = esp_timer_get_time();
    if (!model.run())
      continue;
    Result result{};
    result.seq = ++seq;
    result.decode_us = decode_us;
    result.preprocess_us = preprocessed - started;
    result.invoke_us = model.last_run_us();
    result.delegate_us = model.last_delegate_us();
    top_k(model.logits(), scratch, classes, result.top);
    const int64_t now = esp_timer_get_time();
    result.cycle_us = now - previous;
    result.latency_us = now - captured;
    result.dropped_results = dropped;
    result.decode_errors = decode_errors;
    previous = now;
    if (xQueueSend(g_shared.results, &result, 0) != pdTRUE)
      ++dropped;
    if (PREVIEW) {
      xSemaphoreTake(g_shared.state_lock, portMAX_DELAY);
      uint8_t* destination = g_shared.published_bgr;
      for (size_t i = 0; i < rgb_bytes; i += 3) {
        destination[i] = rgb[i + 2];
        destination[i + 1] = rgb[i + 1];
        destination[i + 2] = rgb[i];
      }
      g_shared.published_ready = true;
      g_shared.published_seq = result.seq;
      xSemaphoreGive(g_shared.state_lock);
    }
    vTaskDelay(1);
  }
}

void send_result(UsbLink& link, const Result& result) {
  char line[352];
  int n = std::snprintf(
      line,
      sizeof(line),
      "#RES seq=%u dec_us=%lld pre_us=%lld infer_us=%lld delegate_us=%lld cycle_us=%lld "
      "latency_us=%lld "
      "dropped=%u dec_err=%u usb_fail=%u top=",
      static_cast<unsigned>(result.seq),
      static_cast<long long>(result.decode_us),
      static_cast<long long>(result.preprocess_us),
      static_cast<long long>(result.invoke_us),
      static_cast<long long>(result.delegate_us),
      static_cast<long long>(result.cycle_us),
      static_cast<long long>(result.latency_us),
      static_cast<unsigned>(result.dropped_results),
      static_cast<unsigned>(result.decode_errors),
      static_cast<unsigned>(link.failed_writes()));
  for (int k = 0; k < TOP_K && n > 0 && n < static_cast<int>(sizeof(line)) - 24;
       ++k) {
    n += std::snprintf(
        line + n,
        sizeof(line) - n,
        "%s%ld:%.4f",
        k ? "," : "",
        static_cast<long>(result.top[k].index),
        static_cast<double>(result.top[k].probability));
  }
  if (n <= 0 || n >= static_cast<int>(sizeof(line)) - 1)
    return;
  line[n++] = '\n';
  link.write_line(line, static_cast<size_t>(n));
}

void send_model_input(
    UsbLink& link,
    JpegLine& line,
    uint8_t* input_copy,
    size_t rgb_bytes) {
  bool have_input = false;
  uint32_t input_seq = 0;
  xSemaphoreTake(g_shared.state_lock, portMAX_DELAY);
  if (g_shared.published_ready) {
    std::memcpy(input_copy, g_shared.published_bgr, rgb_bytes);
    g_shared.published_ready = false;
    input_seq = g_shared.published_seq;
    have_input = true;
  }
  xSemaphoreGive(g_shared.state_lock);
  if (!have_input || !link.host_reading())
    return;
  uint8_t* jpeg = nullptr;
  size_t length = 0;
  const int64_t started = esp_timer_get_time();
  if (fmt2jpg(
          input_copy,
          rgb_bytes,
          static_cast<uint16_t>(g_shared.input_width),
          static_cast<uint16_t>(g_shared.input_height),
          PIXFORMAT_RGB888,
          PREVIEW_QUALITY,
          &jpeg,
          &length)) {
    line.send(
        link,
        line.build(
            "input",
            input_seq,
            input_seq,
            g_shared.input_width,
            g_shared.input_height,
            esp_timer_get_time() - started,
            jpeg,
            length));
  }
  std::free(jpeg);
}

void send_preview_frame(UsbLink& link, JpegLine& line, uint32_t seq) {
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr)
    return;
  const int32_t frame_width = static_cast<int32_t>(frame->width);
  const int32_t frame_height = static_cast<int32_t>(frame->height);
  xSemaphoreTake(g_shared.state_lock, portMAX_DELAY);
  const uint32_t result_seq = g_shared.published_seq;
  xSemaphoreGive(g_shared.state_lock);
  size_t length = 0;
  if (SENSOR_JPEG) {
    length = line.build(
        "cam",
        seq,
        result_seq,
        frame_width,
        frame_height,
        0,
        frame->buf,
        frame->len);
    esp_camera_fb_return(frame);
  } else {
    uint8_t* jpeg = nullptr;
    size_t jpeg_length = 0;
    const int64_t started = esp_timer_get_time();
    const bool encoded = frame2jpg(frame, PREVIEW_QUALITY, &jpeg, &jpeg_length);
    const int64_t encode_us = esp_timer_get_time() - started;
    esp_camera_fb_return(frame);
    if (encoded) {
      length = line.build(
          "cam",
          seq,
          result_seq,
          frame_width,
          frame_height,
          encode_us,
          jpeg,
          jpeg_length);
    }
    std::free(jpeg);
  }
  line.send(link, length);
}

void stream_task(void*) {
  UsbLink link;
  if (!link.begin()) {
    std::printf("#ERR USB-Serial-JTAG driver install failed\n");
    vTaskDelete(nullptr);
    return;
  }
  JpegLine line;
  const size_t rgb_bytes =
      static_cast<size_t>(g_shared.input_width) * g_shared.input_height * 3;
  uint8_t* input_copy =
      PREVIEW ? static_cast<uint8_t*>(psram_alloc(rgb_bytes)) : nullptr;
  if (PREVIEW && (input_copy == nullptr || !line.reserve(64 * 1024))) {
    std::printf("#ERR preview buffers could not be allocated\n");
    vTaskDelete(nullptr);
    return;
  }
  const int64_t period_us = 1000000LL / PREVIEW_MAX_FPS;
  int64_t next_preview = esp_timer_get_time();
  uint32_t preview_seq = 0;
  while (true) {
    TickType_t wait = portMAX_DELAY;
    if (PREVIEW) {
      const int64_t remaining = next_preview - esp_timer_get_time();
      wait = remaining > 0 ? pdMS_TO_TICKS(remaining / 1000) : 0;
    }
    Result result;
    if (xQueueReceive(g_shared.results, &result, wait) == pdTRUE) {
      send_result(link, result);
      if (PREVIEW)
        send_model_input(link, line, input_copy, rgb_bytes);
      continue;
    }
    if (!PREVIEW)
      continue;
    const int64_t now = esp_timer_get_time();
    next_preview = now - next_preview > period_us ? now + period_us
                                                  : next_preview + period_us;
    if (!link.host_reading()) {
      next_preview = now + 500000;
      link.probe();
      continue;
    }
    send_preview_frame(link, line, ++preview_seq);
  }
}

bool start_camera() {
  camera_config_t config{};
  config.pin_pwdn = PINOUT.pwdn;
  config.pin_reset = PINOUT.reset;
  config.pin_xclk = PINOUT.xclk;
  config.pin_sccb_sda = PINOUT.sda;
  config.pin_sccb_scl = PINOUT.scl;
  config.pin_d7 = PINOUT.d7;
  config.pin_d6 = PINOUT.d6;
  config.pin_d5 = PINOUT.d5;
  config.pin_d4 = PINOUT.d4;
  config.pin_d3 = PINOUT.d3;
  config.pin_d2 = PINOUT.d2;
  config.pin_d1 = PINOUT.d1;
  config.pin_d0 = PINOUT.d0;
  config.pin_vsync = PINOUT.vsync;
  config.pin_href = PINOUT.href;
  config.pin_pclk = PINOUT.pclk;
  config.xclk_freq_hz = CONFIG_CAMERA_STREAM_XCLK_MHZ * 1000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = SENSOR_JPEG ? PIXFORMAT_JPEG : PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = SENSOR_JPEG_QUALITY;
  config.fb_count = SENSOR_JPEG ? 2 : 3;
  config.fb_location = SENSOR_JPEG ? CAMERA_FB_IN_DRAM : CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.sccb_i2c_port = 0;
  config.jpeg_buffer_size = SENSOR_JPEG ? SENSOR_JPEG_BUFFER_BYTES : 0;
  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    std::printf(
        "#ERR camera init failed with pinout %s: 0x%x\n",
        PINOUT.name,
        static_cast<unsigned>(error));
    return false;
  }
  sensor_t* sensor = esp_camera_sensor_get();
  const camera_sensor_info_t* info =
      sensor ? esp_camera_sensor_get_info(&sensor->id) : nullptr;
  if (sensor != nullptr) {
    sensor->set_vflip(sensor, VFLIP);
    sensor->set_hmirror(sensor, HMIRROR);
  }
  std::printf(
      "Camera: %s (PID 0x%04x) on pinout %s, QVGA %s, XCLK %d MHz\n",
      info ? info->name : "unknown",
      sensor ? static_cast<unsigned>(sensor->id.PID) : 0U,
      PINOUT.name,
      SENSOR_JPEG ? "JPEG" : "RGB565",
      CONFIG_CAMERA_STREAM_XCLK_MHZ);
  return true;
}

} // namespace

void run_camera_stream() {
  std::printf("\n========================================\n");
  std::printf(" CAMERA INFERENCE STREAM\n");
  std::printf("========================================\n");
  if (SENSOR_JPEG &&
      !probe_decoder_channel_order(g_shared.decoder_swaps_red_blue)) {
    std::printf("#ERR JPEG decoder channel order check failed\n");
    return;
  }
  if (!start_camera())
    return;

  EtModel* model = new (std::nothrow) EtModel();
  const size_t model_size = reinterpret_cast<uintptr_t>(camera_model_end) -
      reinterpret_cast<uintptr_t>(camera_model_start);
  if (model == nullptr || !model->load(camera_model_start, model_size)) {
    std::printf("#ERR model load failed\n");
    return;
  }
  const EtModelInfo& info = model->info();
  g_shared.model = model;
  g_shared.input_width = info.input_width;
  g_shared.input_height = info.input_height;
  g_shared.state_lock = xSemaphoreCreateMutex();
  g_shared.results = xQueueCreate(RESULT_QUEUE_LENGTH, sizeof(Result));
  g_shared.published_bgr = PREVIEW
      ? static_cast<uint8_t*>(psram_alloc(
            static_cast<size_t>(info.input_width) * info.input_height * 3))
      : nullptr;
  if (g_shared.state_lock == nullptr || g_shared.results == nullptr ||
      (PREVIEW && g_shared.published_bgr == nullptr)) {
    std::printf("#ERR stream state could not be allocated\n");
    return;
  }
  std::printf(
      "#CAM model_input=%ldx%ld classes=%u runtime=executorch delegates=%u "
      "delegate_arena=%u planned=%u dual_core=%d preview=%d preview_fps=%d "
      "sensor=%s sensor_quality=%d xclk_mhz=%d\n",
      static_cast<long>(info.input_width),
      static_cast<long>(info.input_height),
      static_cast<unsigned>(info.classes),
      static_cast<unsigned>(info.delegates),
      static_cast<unsigned>(info.delegate_arena_bytes),
      static_cast<unsigned>(info.planned_bytes),
      DUAL_CORE ? 1 : 0,
      PREVIEW ? 1 : 0,
      PREVIEW_MAX_FPS,
      SENSOR_JPEG ? "jpeg" : "rgb565",
      SENSOR_JPEG_QUALITY,
      CONFIG_CAMERA_STREAM_XCLK_MHZ);
  std::printf(
      "Internal RAM free %u (largest %u), PSRAM free %u\n",
      static_cast<unsigned>(
          heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
      static_cast<unsigned>(heap_caps_get_largest_free_block(
          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  xTaskCreatePinnedToCore(
      stream_task, "cam_stream", 8 * 1024, nullptr, 4, nullptr, 0);
  xTaskCreatePinnedToCore(
      inference_task, "cam_infer", 16 * 1024, nullptr, 5, nullptr, 1);
}

} // namespace espressif
} // namespace examples
} // namespace executorch
