/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

bool lcd_init();
void lcd_draw_rgb888(const uint8_t* image, uint32_t width, uint32_t height);
void lcd_draw_detection(
    uint32_t x1,
    uint32_t y1,
    uint32_t x2,
    uint32_t y2,
    uint32_t confidence_percent);
