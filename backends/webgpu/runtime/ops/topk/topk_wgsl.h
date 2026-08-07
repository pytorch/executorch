/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::backends::webgpu {

// @generated from topk.wgsl - DO NOT EDIT.
// wgsl-sha256: 5ee60ff98938deb3a7f531e6bcc9168dce6e356eebfc3f4dc063abfe9caf4437
inline constexpr const char* kTopkWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices_out: array<u32>;
@group(0) @binding(2) var<storage, read> scores: array<u32>;

// The accepted topk.wgsl selects k=32 of 2048 with a 32-entry binary heap on a
// SINGLE invocation (@workgroup_size(1), dispatched 1x1x1). Measured on the W3
// profile at 524.288 us per dispatch = 8 timestamp quanta, i.e. ~256 ns per
// element: one lane, one outstanding global load at a time, latency fully
// exposed because the very next instruction branches on the loaded value.
//
// This kernel changes NOTHING about the algorithm. The heap, the comparator
// and the emission order are transcribed character for character from the
// accepted kernel. The only differences are:
//
//   1. the row is staged into workgroup memory by 64 lanes first, so the
//      serial scan reads `vals[i]` (threadgroup) instead of `scores[i]`
//      (device) -- the loads are hoisted out of the dependent chain and
//      issued 64-wide;
//   2. `heap_values[0]` -- the only heap slot the hot loop reads -- is
//      mirrored into a register `heap_root`, refreshed after every
//      adjust_heap call that can touch slot 0. `heap_values` is dynamically
//      indexed through a pointer, so it is thread-local (device-backed)
//      memory, and without this the hot loop pays a second dependent load per
//      element.
//
// Both are behaviour-preserving by construction, so this kernel is bit-exact
// against the accepted one on EVERY input, including forced ties, NaN
// payloads and signed zeros. In particular it does NOT adopt a
// lowest-index-wins tie rule; that rule is an explicitly killed mutation
// (`tie_by_low_index`) of the accepted CPU authority bundle.

const WG: u32 = 64u;
const N: u32 = 2048u;
const PER_LANE: u32 = N / WG;

var<workgroup> vals: array<u32, N>;

fn is_nan_bits(bits: u32) -> bool {
    return (bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u;
}

fn float_less_than_bits(lhs: u32, rhs: u32) -> bool {
    let lhs_nan = is_nan_bits(lhs);
    let rhs_nan = is_nan_bits(rhs);
    if (lhs_nan || rhs_nan) {
        return !lhs_nan && rhs_nan;
    }

    let lhs_magnitude = lhs & 0x7fffffffu;
    let rhs_magnitude = rhs & 0x7fffffffu;
    if (lhs_magnitude == 0u && rhs_magnitude == 0u) {
        return false;
    }

    let lhs_negative = (lhs & 0x80000000u) != 0u;
    let rhs_negative = (rhs & 0x80000000u) != 0u;
    if (lhs_negative != rhs_negative) {
        return lhs_negative;
    }
    if (lhs_negative) {
        return lhs > rhs;
    }
    return lhs < rhs;
}

fn greater(lhs: u32, rhs: u32) -> bool {
    return float_less_than_bits(rhs, lhs);
}

fn push_heap(
    heap_values: ptr<function, array<u32, 32>>,
    heap_indices: ptr<function, array<u32, 32>>,
    initial_hole: u32,
    top: u32,
    value_bits: u32,
    value_index: u32) {
    var hole = initial_hole;
    while (hole > top) {
        let parent = (hole - 1u) / 2u;
        if (!greater((*heap_values)[parent], value_bits)) {
            break;
        }
        (*heap_values)[hole] = (*heap_values)[parent];
        (*heap_indices)[hole] = (*heap_indices)[parent];
        hole = parent;
    }
    (*heap_values)[hole] = value_bits;
    (*heap_indices)[hole] = value_index;
}

fn adjust_heap(
    heap_values: ptr<function, array<u32, 32>>,
    heap_indices: ptr<function, array<u32, 32>>,
    initial_hole: u32,
    length: u32,
    value_bits: u32,
    value_index: u32) {
    let top = initial_hole;
    var hole = initial_hole;
    var second_child = initial_hole;
    while (second_child < (length - 1u) / 2u) {
        second_child = 2u * (second_child + 1u);
        if (greater(
                (*heap_values)[second_child],
                (*heap_values)[second_child - 1u])) {
            second_child -= 1u;
        }
        (*heap_values)[hole] = (*heap_values)[second_child];
        (*heap_indices)[hole] = (*heap_indices)[second_child];
        hole = second_child;
    }
    if ((length & 1u) == 0u && second_child == (length - 2u) / 2u) {
        second_child = 2u * (second_child + 1u);
        (*heap_values)[hole] = (*heap_values)[second_child - 1u];
        (*heap_indices)[hole] = (*heap_indices)[second_child - 1u];
        hole = second_child - 1u;
    }
    push_heap(
        heap_values,
        heap_indices,
        hole,
        top,
        value_bits,
        value_index);
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    for (var j = 0u; j < PER_LANE; j += 1u) {
        let idx = j * WG + lid.x;
        vals[idx] = scores[idx];
    }
    workgroupBarrier();
    if (lid.x != 0u) {
        return;
    }

    var heap_values: array<u32, 32>;
    var heap_indices: array<u32, 32>;
    for (var i = 0u; i < 32u; i += 1u) {
        heap_values[i] = vals[i];
        heap_indices[i] = i;
    }

    var parent = 15u;
    loop {
        let value_bits = heap_values[parent];
        let value_index = heap_indices[parent];
        adjust_heap(
            &heap_values,
            &heap_indices,
            parent,
            32u,
            value_bits,
            value_index);
        if (parent == 0u) {
            break;
        }
        parent -= 1u;
    }

    var heap_root = heap_values[0];
    for (var i = 32u; i < 2048u; i += 1u) {
        let value_bits = vals[i];
        if (greater(value_bits, heap_root)) {
            adjust_heap(&heap_values, &heap_indices, 0u, 32u, value_bits, i);
            heap_root = heap_values[0];
        }
    }

    var last = 32u;
    while (last > 1u) {
        last -= 1u;
        let value_bits = heap_values[last];
        let value_index = heap_indices[last];
        heap_values[last] = heap_values[0];
        heap_indices[last] = heap_indices[0];
        adjust_heap(
            &heap_values,
            &heap_indices,
            0u,
            last,
            value_bits,
            value_index);
    }

    for (var i = 0u; i < 32u; i += 1u) {
        values_out[i] = heap_values[i];
        indices_out[i] = heap_indices[i];
    }
}
)";

inline constexpr uint32_t kTopkWorkgroupSizeX = 64;
inline constexpr uint32_t kTopkWorkgroupSizeY = 1;
inline constexpr uint32_t kTopkWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
