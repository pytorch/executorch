/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

var Module = {};
const et = Module;
beforeAll((done) => {
    et.onRuntimeInitialized = () => {
        done();
    }
});

describe("Tensor", () => {
    test("ones", () => {
        const tensor = et.Tensor.ones([2, 2]);
        expect(tensor.data).toEqual(new Float32Array([1, 1, 1, 1]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("zeros", () => {
        const tensor = et.Tensor.zeros([2, 2]);
        expect(tensor.data).toEqual(new Float32Array([0, 0, 0, 0]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("fromArray", () => {
        const tensor = et.Tensor.fromArray([2, 2], [1, 2, 3, 4]);
        expect(tensor.data).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("fromGenerator", () => {
        function* generator() {
            yield* [1, 2, 3, 4];
        }
        const tensor = et.Tensor.fromIter([2, 2], generator());
        expect(tensor.data).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("fromArray wrong size", () => {
        expect(() => et.Tensor.fromArray([3, 2], [1, 2, 3, 4])).toThrow();
    });

    test("full", () => {
        const tensor = et.Tensor.full([2, 2], 3);
        expect(tensor.data).toEqual(new Float32Array([3, 3, 3, 3]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("scalar type", () => {
        const tensor = et.Tensor.ones([2, 2]);
        expect(tensor.scalarType).toEqual(et.ScalarType.Float);
        tensor.delete();
    });

    test("long tensor", () => {
        const tensor = et.Tensor.ones([2, 2], et.ScalarType.Long);
        expect(tensor.data).toEqual(new BigInt64Array([1n, 1n, 1n, 1n]));
        expect(tensor.sizes).toEqual([2, 2]);
        expect(tensor.scalarType).toEqual(et.ScalarType.Long);
        tensor.delete();
    });

    test("infer long tensor", () => {
        // Number cannot be converted to Long, so we use BigInt instead.
        const tensor = et.Tensor.fromArray([2, 2], [1n, 2n, 3n, 4n]);
        expect(tensor.data).toEqual(new BigInt64Array([1n, 2n, 3n, 4n]));
        expect(tensor.sizes).toEqual([2, 2]);
        expect(tensor.scalarType).toEqual(et.ScalarType.Long);
        tensor.delete();
    });

    test("with dim order and strides", () => {
        const tensor = et.Tensor.fromArray([2, 2], [1, 2, 3, 4], et.ScalarType.Float, [0, 1], [2, 1]);
        expect(tensor.data).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(tensor.sizes).toEqual([2, 2]);
        tensor.delete();
    });

    test("incorrect dim order", () => {
        expect(() => et.Tensor.fromArray([2, 2], [1, 2, 3, 4], et.ScalarType.Float, [1])).toThrow();
        expect(() => et.Tensor.fromArray([2, 2], [1, 2, 3, 4], et.ScalarType.Float, [1, 2])).toThrow();
    });

    test("incorrect strides", () => {
        expect(() => et.Tensor.fromArray([2, 2], [1, 2, 3, 4], et.ScalarType.Float, [1, 1], [2, 1])).toThrow();
    });
});

describe("Module", () => {
    test("getMethods has foward", () => {
        const module = et.Module.load("add.pte");
        const methods = module.getMethods();
        expect(methods).toEqual(["forward"]);
        module.delete();
    });

    test("multiple methods", () => {
        const module = et.Module.load("test.pte");
        const methods = module.getMethods();
        expect(methods).toEqual(expect.arrayContaining(["forward", "index", "add_all"]));
        module.delete();
    });

    test("loadMethod forward", () => {
        const module = et.Module.load("add.pte");
        expect(() => module.loadMethod("forward")).not.toThrow();
        module.delete();
    });

    test("loadMethod does not exist", () => {
        const module = et.Module.load("add.pte");
        expect(() => module.loadMethod("does_not_exist")).toThrow();
        module.delete();
    });

    test("load from Uint8Array", () => {
        const data = FS.readFile('add.pte');
        const module = et.Module.load(data);
        const methods = module.getMethods();
        expect(methods).toEqual(["forward"]);
        module.delete();
    });

    test("load from ArrayBuffer", () => {
        const data = FS.readFile('add.pte');
        const module = et.Module.load(data.buffer);
        const methods = module.getMethods();
        expect(methods).toEqual(["forward"]);
        module.delete();
    });

    describe("MethodMeta", () => {
        test("name is forward", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.name).toEqual("forward");
            module.delete();
        });

        test("inputs are tensors", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.inputTags.length).toEqual(3);
            expect(methodMeta.inputTags).toEqual([et.Tag.Tensor, et.Tag.Tensor, et.Tag.Tensor]);
            module.delete();
        });

        test("outputs are tensors", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.outputTags.length).toEqual(1);
            expect(methodMeta.outputTags).toEqual([et.Tag.Tensor]);
            module.delete();
        });

        test("num instructions is 2", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.numInstructions).toEqual(2);
            module.delete();
        });

        test("method does not exist", () => {
            const module = et.Module.load("add_mul.pte");
            expect(() => module.getMethodMeta("does_not_exist")).toThrow();
            module.delete();
        });

        describe("TensorInfo", () => {
            test("input sizes is 2x2", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                expect(methodMeta.inputTensorMeta.length).toEqual(3);
                methodMeta.inputTensorMeta.forEach((tensorInfo) => {
                    expect(tensorInfo.sizes).toEqual([2, 2]);
                });
                module.delete();
            });

            test("output sizes is 2x2", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                expect(methodMeta.outputTensorMeta.length).toEqual(1);
                expect(methodMeta.outputTensorMeta[0].sizes).toEqual([2, 2]);
                module.delete();
            });

            test("dim order is contiguous", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                methodMeta.inputTensorMeta.forEach((tensorInfo) => {
                    expect(tensorInfo.dimOrder).toEqual([0, 1]);
                });
                module.delete();
            });

            test("scalar type is float", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                methodMeta.inputTensorMeta.forEach((tensorInfo) => {
                    expect(tensorInfo.scalarType).toEqual(et.ScalarType.Float);
                });
                module.delete();
            });

            test("memory planned", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                methodMeta.inputTensorMeta.forEach((tensorInfo) => {
                    expect(tensorInfo.isMemoryPlanned).toBe(true);
                });
                module.delete();
            });

            test("nbytes is 16", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                methodMeta.inputTensorMeta.forEach((tensorInfo) => {
                    expect(tensorInfo.nbytes).toEqual(16);
                });
                module.delete();
            });

            test("non-tensor in input", () => {
                const module = et.Module.load("test.pte");
                const methodMeta = module.getMethodMeta("add_all");
                expect(methodMeta.inputTags).toEqual([et.Tag.Tensor, et.Tag.Int]);
                expect(methodMeta.inputTensorMeta[0]).not.toBeUndefined();
                expect(methodMeta.inputTensorMeta[1]).toBeUndefined();
                module.delete();
            });

            test("non-tensor in output", () => {
                const module = et.Module.load("test.pte");
                const methodMeta = module.getMethodMeta("add_all");
                expect(methodMeta.outputTags).toEqual([et.Tag.Tensor, et.Tag.Int, et.Tag.Tensor]);
                expect(methodMeta.outputTensorMeta[0]).not.toBeUndefined();
                expect(methodMeta.outputTensorMeta[1]).toBeUndefined();
                expect(methodMeta.outputTensorMeta[2]).not.toBeUndefined();
                module.delete();
            });
        });
    });

    describe("execute", () => {
        test("add normally", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.Tensor.ones([1]), et.Tensor.ones([1])];
            const output = module.execute("forward", inputs);

            expect(output.length).toEqual(1);
            expect(output[0].data).toEqual(new Float32Array([2]));
            expect(output[0].sizes).toEqual([1]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("add_mul normally", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2])];
            const output = module.execute("forward", inputs);

            expect(output.length).toEqual(1);
            expect(output[0].data).toEqual(new Float32Array([3, 3, 3, 3]));
            expect(output[0].sizes).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("forward directly", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2])];
            const output = module.forward(inputs);

            expect(output.length).toEqual(1);
            expect(output[0].data).toEqual(new Float32Array([3, 3, 3, 3]));
            expect(output[0].sizes).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("wrong number of inputs", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2])];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("wrong input size", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.Tensor.ones([2, 1]), et.Tensor.ones([2, 1])];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("wrong input type", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.Tensor.ones([1]), et.Tensor.ones([1], et.ScalarType.Long)];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("method does not exist", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.Tensor.ones([1]), et.Tensor.ones([1])];
            expect(() => module.execute("does_not_exist", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("output tensor can be reused", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2]), et.Tensor.ones([2, 2])];
            const output = module.forward(inputs);

            expect(output.length).toEqual(1);
            expect(output[0].data).toEqual(new Float32Array([3, 3, 3, 3]));
            expect(output[0].sizes).toEqual([2, 2]);

            const inputs2 = [output[0], output[0], output[0]];
            const output2 = module.forward(inputs2);

            expect(output2.length).toEqual(1);
            expect(output2[0].data).toEqual(new Float32Array([21, 21, 21, 21]));
            expect(output2[0].sizes).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            output2.forEach((output) => output.delete());
            module.delete();
        });
    });
});

describe("sanity", () => {
    // Emscripten enums are equal by default for some reason.
    test("different enums are not equal", () => {
        expect(et.ScalarType.Float).not.toEqual(et.ScalarType.Long);
        expect(et.Tag.Int).not.toEqual(et.Tag.Double);
    });
});
