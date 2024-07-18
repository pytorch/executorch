/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch::executor::native {

template <typename T>
void randomized_fast_hadamard_transform_impl(
    const T* vec,
    T* out,
    const std::uint8_t* randomization_bitvec,
    int vec_size);

// Compute the fast Walsh-Hadamard transform
// (https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
// of vec, randomized using the additional bitvector randomization_bitvec.
// randomization_bitvec is a bitvector interpreted as a vector containing only
// elements from the set {+1, -1} (where the bit 1 denotes -1 and the bit 0
// denotes +1
// -- this choice of convention allows straightforward efficient
// implementation). Whereas the usual FWHT computes H @ vec for some
// Hadamard matrix H, the randomized FWHT computes (H @ diag(s)) @
// vec. (Note that the multiplication by diag(s) is equivalent to
// flipping the sign bit for each output element where s contains a 1
// bit in the corresponding position.)
//
// vec.size() is currently required to be a power of two.
Tensor& randomized_fast_hadamard_transform_out(
    RuntimeContext& ctx,
    const Tensor& vec,
    const Tensor& randomization_bitvec,
    Tensor& out);
} // namespace torch::executor::native
