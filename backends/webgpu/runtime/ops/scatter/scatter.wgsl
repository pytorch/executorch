// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;

@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 4096u; i += 1u) {
        let destination = indices[i];
        if (destination >= 0 && destination < 262144) {
            output[u32(destination)] = source[i];
        }
    }
}
