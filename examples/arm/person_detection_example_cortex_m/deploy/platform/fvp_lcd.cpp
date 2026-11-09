/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fvp_lcd.h"

#include <algorithm>

namespace {

constexpr uintptr_t kLcdBase = 0x5930A000;
auto* const command = reinterpret_cast<volatile uint32_t*>(kLcdBase);
auto* const data = reinterpret_cast<volatile uint32_t*>(kLcdBase + 0x004);
auto* const misc = reinterpret_cast<volatile uint32_t*>(kLcdBase + 0x04C);

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint16_t kRed = 0xF800;
constexpr uint16_t kBlack = 0x0000;
constexpr uint16_t kWhite = 0xFFFF;
constexpr uint32_t kChipSelect = 1U << 0;
constexpr uint32_t kReset = 1U << 3;
constexpr uint32_t kBacklight = 1U << 6;

void delay(uint32_t count) {
  for (volatile uint32_t i = 0; i < (count << 8); ++i) {
  }
}

void select(bool enabled) {
  if (enabled) {
    *misc &= ~kChipSelect;
  } else {
    *misc |= kChipSelect;
  }
}

void write_data(uint16_t value) {
  *data = value >> 8;
  // Both writes are required transfers to the volatile LCD data register.
  // cppcheck-suppress redundantAssignment
  *data = value & 0xFF;
}

void write_register(uint8_t reg, uint16_t value) {
  select(true);
  *command = reg;
  write_data(value);
  select(false);
}

void set_window(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
  const uint32_t xe = x + width - 1;
  const uint32_t ye = y + height - 1;
  write_register(0x02, x >> 8);
  write_register(0x03, x & 0xFF);
  write_register(0x04, xe >> 8);
  write_register(0x05, xe & 0xFF);
  write_register(0x06, y >> 8);
  write_register(0x07, y & 0xFF);
  write_register(0x08, ye >> 8);
  write_register(0x09, ye & 0xFF);
}

void begin_pixels() {
  select(true);
  *command = 0x22;
}

void end_pixels() {
  select(false);
}

void fill_rect(
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    uint16_t color) {
  if (x >= kWidth || y >= kHeight || width == 0 || height == 0) {
    return;
  }
  width = std::min(width, kWidth - x);
  height = std::min(height, kHeight - y);
  set_window(x, y, width, height);
  begin_pixels();
  for (uint32_t i = 0; i < width * height; ++i) {
    write_data(color);
  }
  end_pixels();
}

const uint8_t* glyph(char character) {
  static constexpr uint8_t digits[10][7] = {
      {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
      {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
      {0x0E, 0x11, 0x10, 0x08, 0x04, 0x02, 0x1F},
      {0x1F, 0x08, 0x04, 0x08, 0x10, 0x11, 0x0E},
      {0x08, 0x0C, 0x0A, 0x09, 0x1F, 0x08, 0x08},
      {0x1F, 0x01, 0x0F, 0x10, 0x10, 0x11, 0x0E},
      {0x0C, 0x02, 0x01, 0x0F, 0x11, 0x11, 0x0E},
      {0x1F, 0x10, 0x08, 0x04, 0x02, 0x02, 0x02},
      {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
      {0x0E, 0x11, 0x11, 0x1E, 0x10, 0x08, 0x06},
  };
  static constexpr uint8_t percent[7] = {
      0x19, 0x19, 0x08, 0x04, 0x02, 0x13, 0x13};
  if (character >= '0' && character <= '9') {
    return digits[character - '0'];
  }
  return percent;
}

void draw_character(uint32_t x, uint32_t y, char character) {
  const uint8_t* rows = glyph(character);
  for (uint32_t row = 0; row < 7; ++row) {
    for (uint32_t column = 0; column < 5; ++column) {
      if ((rows[row] & (1U << (4 - column))) != 0) {
        fill_rect(x + column, y + row, 1, 1, kWhite);
      }
    }
  }
}

} // namespace

bool lcd_init() {
  *misc |= kChipSelect | kReset;
  *misc &= ~kBacklight;
  *misc &= ~kReset;
  delay(1);
  *misc |= kReset;
  delay(10);

  const struct {
    uint8_t reg;
    uint16_t value;
  } setup[] = {
      {0xEA, 0x00},     {0xEB, 0x20},       {0xEC, 0x0C}, {0xED, 0xC7},
      {0xE8, 0x38},     {0xE9, 0x10},       {0xF1, 0x01}, {0xF2, 0x10},
      {0x1B, 0x1B},     {0x1A, 0x01},       {0x24, 0x2F}, {0x25, 0x57},
      {0x23, 0x88},     {0x18, 0x36},       {0x19, 0x01}, {0x01, 0x00},
      {0x17, 0x55},     {0x00, 0x00},       {0x2F, 0x11}, {0x31, 0x00},
      {0x32, 0x00},     {0x16, 0x68},       {0x0E, 0x00}, {0x0F, 0x00},
      {0x10, 320 >> 8}, {0x11, 320 & 0xFF}, {0x12, 0x00}, {0x13, 0x00},
  };
  for (const auto& setting : setup) {
    write_register(setting.reg, setting.value);
  }
  write_register(0x1F, 0x88);
  delay(20);
  write_register(0x1F, 0x82);
  delay(5);
  write_register(0x1F, 0x92);
  delay(5);
  write_register(0x1F, 0xD2);
  delay(5);
  write_register(0x28, 0x38);
  delay(20);
  write_register(0x28, 0x3C);
  *misc |= kBacklight;
  fill_rect(0, 0, kWidth, kHeight, kBlack);
  return true;
}

void lcd_draw_rgb888(const uint8_t* image, uint32_t width, uint32_t height) {
  if (image == nullptr || width != kWidth || height != kHeight) {
    return;
  }
  set_window(0, 0, width, height);
  begin_pixels();
  for (uint32_t i = 0; i < width * height; ++i) {
    const uint8_t red = image[i * 3];
    const uint8_t green = image[i * 3 + 1];
    const uint8_t blue = image[i * 3 + 2];
    write_data(((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3));
  }
  end_pixels();
}

void lcd_draw_detection(
    uint32_t x1,
    uint32_t y1,
    uint32_t x2,
    uint32_t y2,
    uint32_t confidence_percent) {
  x1 = std::min(x1, kWidth - 1);
  x2 = std::min(x2, kWidth - 1);
  y1 = std::min(y1, kHeight - 1);
  y2 = std::min(y2, kHeight - 1);
  if (x2 <= x1 || y2 <= y1) {
    return;
  }
  fill_rect(x1, y1, x2 - x1 + 1, 2, kRed);
  fill_rect(x1, y2 - 1, x2 - x1 + 1, 2, kRed);
  fill_rect(x1, y1, 2, y2 - y1 + 1, kRed);
  fill_rect(x2 - 1, y1, 2, y2 - y1 + 1, kRed);

  confidence_percent = std::min(confidence_percent, static_cast<uint32_t>(100));
  const uint32_t label_x = x1;
  const uint32_t label_y = y1 > 9 ? y1 - 9 : y1 + 3;
  const bool three_digits = confidence_percent == 100;
  const uint32_t characters = three_digits ? 4 : 3;
  fill_rect(label_x, label_y, characters * 6, 9, kBlack);
  uint32_t position = label_x + 1;
  if (three_digits) {
    draw_character(position, label_y + 1, '1');
    position += 6;
    draw_character(position, label_y + 1, '0');
    position += 6;
  } else {
    draw_character(position, label_y + 1, '0' + confidence_percent / 10);
    position += 6;
  }
  draw_character(position, label_y + 1, '0' + confidence_percent % 10);
  position += 6;
  draw_character(position, label_y + 1, '%');
}
