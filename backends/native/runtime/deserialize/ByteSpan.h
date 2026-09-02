// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <span>

namespace ptn {

// A borrowed, read-only view of a byte range. The package readers hand these
// out instead of copying: every span in a loaded Package aliases the one buffer
// the Package owns, and is valid only for that Package's lifetime.
//
// Named rather than spelled out at each use so that contract has somewhere to
// live; it is a plain std::span otherwise.
using ByteSpan = std::span<const uint8_t>;

} // namespace ptn
