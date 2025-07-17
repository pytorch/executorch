
let et;
beforeAll((done) => {
    et = require("./executorch_wasm_test_lib");
    et.onRuntimeInitialized = () => {
        done();
    }
});

describe("Tensor", () => {
    test("ones", () => {
        const tensor = et.FloatTensor.ones([2, 2]);
        expect(tensor.getData()).toEqual([1, 1, 1, 1]);
        expect(tensor.getSizes()).toEqual([2, 2]);
        tensor.delete();
    });

    test("zeros", () => {
        const tensor = et.FloatTensor.zeros([2, 2]);
        expect(tensor.getData()).toEqual([0, 0, 0, 0]);
        expect(tensor.getSizes()).toEqual([2, 2]);
        tensor.delete();
    });

    test("fromArray", () => {
        const tensor = et.FloatTensor.fromArray([1, 2, 3, 4], [2, 2]);
        expect(tensor.getData()).toEqual([1, 2, 3, 4]);
        expect(tensor.getSizes()).toEqual([2, 2]);
        tensor.delete();
    });

    test("fromArray wrong size", () => {
        expect(() => et.FloatTensor.fromArray([1, 2, 3, 4], [3, 2])).toThrow();
    });

    test("full", () => {
        const tensor = et.FloatTensor.full([2, 2], 3);
        expect(tensor.getData()).toEqual([3, 3, 3, 3]);
        expect(tensor.getSizes()).toEqual([2, 2]);
        tensor.delete();
    });
});

describe("Module", () => {
    test("getMethods has foward", () => {
        const module = et.Module.load("add.pte");
        const methods = module.getMethods();
        expect(methods).toEqual(["forward"]);
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

    describe("MethodMeta", () => {
        test("name is forward", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.name).toEqual("forward");
            methodMeta.delete();
            module.delete();
        });

        test("numInputs is 3", () => {
            const module = et.Module.load("add_mul.pte");
            const methodMeta = module.getMethodMeta("forward");
            expect(methodMeta.numInputs).toEqual(3);
            methodMeta.delete();
            module.delete();
        });

        test("method does not exist", () => {
            const module = et.Module.load("add_mul.pte");
            expect(() => module.getMethodMeta("does_not_exist")).toThrow();
            module.delete();
        });

        describe("TensorInfo", () => {
            test("sizes is 2x2", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                for (var i = 0; i < methodMeta.numInputs; i++) {
                    const tensorInfo = methodMeta.inputTensorMeta(i);
                    expect(tensorInfo.sizes).toEqual([2, 2]);
                    tensorInfo.delete();
                }
                methodMeta.delete();
                module.delete();
            });

            test("out of range", () => {
                const module = et.Module.load("add_mul.pte");
                const methodMeta = module.getMethodMeta("forward");
                expect(() => methodMeta.inputTensorMeta(3)).toThrow();
                methodMeta.delete();
                module.delete();
            });
        });
    });

    describe("execute", () => {
        test("add normally", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.FloatTensor.ones([1]), et.FloatTensor.ones([1])];
            const output = module.execute("forward", inputs);

            expect(output.length).toEqual(1);
            expect(output[0].getData()).toEqual([2]);
            expect(output[0].getSizes()).toEqual([1]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("add_mul normally", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2])];
            const output = module.execute("forward", inputs);

            expect(output.length).toEqual(1);
            expect(output[0].getData()).toEqual([3, 3, 3, 3]);
            expect(output[0].getSizes()).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("forward directly", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2])];
            const output = module.forward(inputs);

            expect(output.length).toEqual(1);
            expect(output[0].getData()).toEqual([3, 3, 3, 3]);
            expect(output[0].getSizes()).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            module.delete();
        });

        test("wrong number of inputs", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2])];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("wrong input size", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.FloatTensor.ones([2, 1]), et.FloatTensor.ones([2, 1])];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("wrong input type", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.FloatTensor.ones([1]), et.IntTensor.ones([1])];
            expect(() => module.execute("forward", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("method does not exist", () => {
            const module = et.Module.load("add.pte");
            const inputs = [et.FloatTensor.ones([1]), et.FloatTensor.ones([1])];
            expect(() => module.execute("does_not_exist", inputs)).toThrow();

            inputs.forEach((input) => input.delete());
            module.delete();
        });

        test("output tensor can be reused", () => {
            const module = et.Module.load("add_mul.pte");
            const inputs = [et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2]), et.FloatTensor.ones([2, 2])];
            const output = module.forward(inputs);

            expect(output.length).toEqual(1);
            expect(output[0].getData()).toEqual([3, 3, 3, 3]);
            expect(output[0].getSizes()).toEqual([2, 2]);

            const inputs2 = [output[0], output[0], output[0]];
            const output2 = module.forward(inputs2);

            expect(output2.length).toEqual(1);
            expect(output2[0].getData()).toEqual([21, 21, 21, 21]);
            expect(output2[0].getSizes()).toEqual([2, 2]);

            inputs.forEach((input) => input.delete());
            output.forEach((output) => output.delete());
            output2.forEach((output) => output.delete());
            module.delete();
        });
    });
});
