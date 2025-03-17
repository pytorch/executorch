/*
llama.cpp - commit 54ef9cfc
https://github.com/ggerganov/llama.cpp

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct range_nfd {
  uint32_t first;
  uint32_t last;
  uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>>
    unicode_ranges_flags;

constexpr std::array<uint32_t, 25> unicode_set_whitespace = {
    0x000009, 0x00000A, 0x00000B, 0x00000C, 0x00000D, 0x000020, 0x000085,
    0x0000A0, 0x001680, 0x002000, 0x002001, 0x002002, 0x002003, 0x002004,
    0x002005, 0x002006, 0x002007, 0x002008, 0x002009, 0x00200A, 0x002028,
    0x002029, 0x00202F, 0x00205F, 0x003000,
};

extern const std::initializer_list<std::pair<uint32_t, uint32_t>>
    unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>>
    unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;
