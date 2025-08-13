/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

describe("ETDump", () => {
    test("etdump enabled", () => {
        const module = et.Module.load("add_mul.pte");
        const inputs = [et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2])];
        const output = module.forward(inputs);

        inputs.forEach((input) => input.delete());
        output.forEach((output) => output.delete());
        const etdump = module.etdump();
        const buffer = etdump.buffer;
        expect(buffer).toBeInstanceOf(Uint8Array);
        expect(buffer.length).toBeGreaterThan(0);
        etdump.delete();
        module.delete();
    });
});
