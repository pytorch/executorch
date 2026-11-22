// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// REFERENCE decoder for the GGUF PackedQuant codecs "gguf:q4_0", "gguf:q5_0",
// "gguf:q5_1", "gguf:q8_0", "gguf:q2_k", "gguf:q3_k", "gguf:q4_k", "gguf:q5_k"
// and "gguf:q6_k" (blocks in ggml's layout for the type the codec names, see
// the README), producing the fp32 value a PackedQuant constant denotes.
//
// It decodes as ggml's dequantization does, and is a temporary fallback for
// engines that cannot lower a codec yet. It is not a performance path and may
// be removed: decoding expands the weights to fp32 on the host and on the
// device. Engines are expected to lower each PackedQuant codec natively.
namespace ptn::reference {

bool can_decode_gguf(std::string_view codec);

// Stored bytes of one row of `cols` weights. Throws std::invalid_argument for
// a codec the decoder does not handle or a `cols` that is not a positive
// multiple of the codec's block size.
size_t gguf_row_nbytes(std::string_view codec, int64_t cols);

// Decodes a [rows, cols] constant of `nbytes` stored bytes into
// out[0, rows * cols), spreading rows over at most `max_threads` threads (0
// uses the hardware concurrency). Throws std::invalid_argument if `nbytes` is
// not rows * gguf_row_nbytes(codec, cols).
void decode_gguf(
    std::string_view codec,
    const uint8_t* data,
    size_t nbytes,
    int64_t rows,
    int64_t cols,
    float* out,
    unsigned max_threads = 0);

} // namespace ptn::reference
