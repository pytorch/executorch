
const et = require("./executorch_wasm");
et.onRuntimeInitialized = () => {
    module = et.Module.load("model.pte");
    var methods = module.getMethods();
    console.log(methods);
    var method = methods[0];
    var methodMeta = module.getMethodMeta(method);
    var inputs = [];
    for (var i = 0; i < methodMeta.numInputs; i++) {
        var tensor = et.FloatTensor.ones(methodMeta.inputTensorMeta(i).sizes);
        console.log("input", i, tensor.getData(), tensor.getSizes());
        inputs.push(tensor);
    }
    var output = module.execute(method, inputs);

    for (var i = 0; i < inputs.length; i++) {
        inputs[i].delete();
    }

    for (var i = 0; i < output.length; i++) {
        console.log("output", i, output[i].getData(), output[i].getSizes());
        output[i].delete();
    }

    module.delete();
}
