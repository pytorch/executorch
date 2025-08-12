# ExecuTorch Wasm Extension

This directory contains the source code for the ExecuTorch Wasm extension. The extension is a C++ library that provides a JavaScript API for ExecuTorch models. The extension is compiled to WebAssembly and can be used in JavaScript applications.

## Installing Emscripten

[Emscripten](https://emscripten.org/index.html) is necessary to compile ExecuTorch for Wasm. You can install Emscripten with these commands:

```bash
# Clone the emsdk repository
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Download and install version 4.0.10 of the SDK
./emsdk install 4.0.10
./emsdk activate 4.0.10

# Add the Emscripten environment variables to your shell
source ./emsdk_env.sh
```

## Building ExecuTorch for Wasm

To build ExecuTorch for Wasm, make sure to use the `emcmake cmake` command and to have `EXECUTORCH_BUILD_WASM` enabled. For example:

```bash
# Configure the build with the Emscripten environment variables
emcmake cmake . -DEXECUTORCH_BUILD_WASM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-out-wasm

# Build the Wasm extension
cmake --build cmake-out-wasm --target executorch_wasm -j32
```

To reduce the binary size, you may also use the selective build options found in the [Kernel Library Selective Build guide](../../docs/source/kernel-library-selective-build.md). You may also use optimized kernels with the `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` option. Portable kernels are used by default.

### Building for Web

In your CMakeLists.txt, add the following lines:

```cmake
add_executable(executorch_wasm_lib) # Emscripten outputs this as a JS and Wasm file
target_link_libraries(executorch_wasm_lib PRIVATE executorch_wasm)
target_link_options(executorch_wasm_lib PRIVATE ...) # Add any additional link options here
```

You can find the Emscripten link options in the [emcc reference](https://emscripten.org/docs/tools_reference/emcc.html).

Building this should output `executorch_wasm_lib.js` and `executorch_wasm_lib.wasm` in the build directory. You can then use this file in your page.

```html
<script>
  // Emscripten calls Module.onRuntimeInitialized once the runtime is ready.
  var Module = {
    onRuntimeInitialized: function() {
      const et = Module; // Assign Module into et for ease of use
      const model = et.Module.load("mv2.pte");
      // ...
    }
  }
</script>
<script src="executorch_wasm_lib.js"></script>
```

### Building for Node.js

While the standard way to import a module in Node.js is to use the `require` function, doing so does not give you access to the [Emscripten API](https://emscripten.org/docs/api_reference/index.html) which would be stored in the globals. For example, you may want to use the [File System API](https://emscripten.org/docs/api_reference/Filesystem-API.html) in your unit tests, which cannot be done if the library is loaded with `require`. Instead, you can use the `--pre-js` option to prepend your file to the start of the JS output and behave similarly to the example in the [Web build](#building-for-web).

```cmake
add_executable(my_project) # Emscripten outputs this as a JS and Wasm file
target_link_libraries(my_project PRIVATE executorch_wasm)
target_link_options(my_project PRIVATE --pre-js my_code.js) # Add any additional link options here
```

The output `my_project.js` should contain both the emitted JS code and the contents of `my_code.js` prepended.

## JavaScript API

### Module
- `static load(data)`: Load a model from a file or a buffer.
- `getMethods()`: Returns the list of methods in the model.
- `loadMethod(methodName)`: Load a method from the model.
- `getMethodMetadata(methodName)`: Get the metadata of a method.
- `etdump()`: If enabled, flushes the etdump buffer and return the results.
- `execute(methodName, inputs)`: Execute a method with the given inputs.
- `forward(inputs)`: Execute the forward method with the given inputs.
- `delete()`: Delete the model from memory.

### Tensor
- `static zeroes(shape, dtype=ScalarType.Float)`: Create a tensor of zeros with the given shape and dtype.
- `static ones(shape, dtype=ScalarType.Float)`: Create a tensor of ones with the given shape and dtype.
- `static full(shape, value, dtype=ScalarType.Float)`: Create a tensor of the given value with the given shape and dtype
- `static fromArray(shape, array, dtype=ScalarType.Float, dimOrder=[], strides=[])`: Create a tensor from a JavaScript array.
- `static fromIter(shape, iter, dtype=ScalarType.Float, dimOrder=[], strides=[])`: Create a tensor from an iterable.
- `delete()`: Delete the tensor from memory.
- `scalarType`: The scalar type of the tensor.
- `data`: The data buffer of the tensor.
- `sizes`: The sizes of the tensor.

### MethodMeta
- `name`: The name of the method.
- `inputTags`: The input tags of the method.
- `inputTensorMeta`: The input tensor metadata of the method.
- `outputTags`: The output tags of the method.
- `outputTensorMeta`: The output tensor metadata of the method.
- `attributeTensorMeta`: The attribute tensor metadata of the method.
- `memoryPlannedBufferSizes`: The memory planned buffer sizes of the method.
- `backends`: The backends of the method.
- `numInstructions`: The number of instructions in the method.
- These are value types and do not need to be manually deleted.

### TensorInfo
- `sizes`: The sizes of the tensor.
- `dimOrder`: The dimension order of the tensor.
- `scalarType`: The scalar type of the tensor.
- `isMemoryPlanned`: Whether the tensor is memory planned.
- `nBytes`: The number of bytes in the tensor.
- `name`: The name of the tensor.
- These are value types and do not need to be manually deleted.

### ETDumpResult
- `buffer`: The buffer containing the ETDump data.
- `delete()`: Delete the ETDumpResult from memory.

### ScalarType
- Only `Float` and `Long` are currently supported.
- `value`: The int constant value of the enum.
- `name`: The `ScalarType` as a string.

### Tag
- `value`: The int constant value of the enum.
- `name`: The `Tag` as a string.

Emscripten's JavaScript API is also avaiable, which you can find more information about it in their [API Reference](https://emscripten.org/docs/api_reference/index.html).
