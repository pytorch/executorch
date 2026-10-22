// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <span>

namespace ptn {

// Borrowed byte-range views. The producer defines their lifetime.
using ByteSpan = std::span<const uint8_t>;
using MutableByteSpan = std::span<uint8_t>;

} // namespace ptn
