# Tokenizers JavaScript Bindings

This directory contains the JavaScript bindings for the [LLM Tokenizers](../../llm/README.md#tokenizer) library.

## Building

To build Tokenizers for Wasm, make sure to use the `emcmake cmake` command and to have `EXECUTORCH_BUILD_TOKENIZERS_WASM` and `EXECUTORCH_BUILD_EXTENSION_LLM` enabled. For example:

```bash
# Configure the build with the Emscripten environment variables
emcmake cmake . -DEXECUTORCH_BUILD_TOKENIZERS_WASM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-out-wasm

# Build the Wasm extension
cmake --build cmake-out-wasm --target tokenizers_wasm -j32
```

Emscripten modules are loaded into the global `Module` object by default. This means you cannot have multiple modules in the same page. If you are also using the ExecuTorch Wasm bindings, it is recommended to use the `MODULARIZE` option to avoid conflicts.

In your CMakeLists.txt, add the following lines:

```cmake
add_executable(tokenizers_wasm_lib) # Emscripten outputs this as a JS and Wasm file
target_link_libraries(tokenizers_wasm_lib PRIVATE tokenizers_wasm)
target_link_options(tokenizers_wasm_lib PRIVATE -sMODULARIZE=1 -sEXPORT_NAME=loadTokenizers) # If EXPORT_NAME is not set, the default is Module, which will conflict with ExecuTorch
```

You can then access the module with `mod = await loadTokenizers();` or `loadTokenizers().then(mod => { /* ... */ });`.

For example, to load the module in a HTML file, you can use the following:

```html
<script src="tokenizers_wasm_lib.js"></script>
<script>
  var Module = {
    onRuntimeInitialized: async function() {
      // Load Tokenizers Module after ExecuTorch Module is initialized
      const tokenizersModule = await loadTokenizers();
      const sp = new tokenizersModule.SpTokenizer();
      // ...
    }
  }
</script>
<script src="executorch_wasm_lib.js"></script>
```

You can read more about Modularized Output in the [Emscripten docs](https://emscripten.org/docs/compiling/Modularized-Output.html).

## JavaScript API

### Supported Tokenizers
- `HFTokenizer`
- `SpTokenizer`
- `Tiktoken`
- `Llama2cTokenizer`

### Tokenizer API
- `load(data)`: Load tokenizer data from a file or a buffer.
- `encode(text, bos=0, eos=0)`: Encode a string into a list of tokens with the number of bos tokens to prepend and eos tokens to append to the result.
- `decode(tokens)`: Decode a list of tokens into a string.
- `vocabSize`: The number of tokens in the vocabulary.
- `eosTok`: The end-of-sequence token.
- `bosTok`: The begining-of-sequence token.
- `isLoaded`: Whether the tokenizer is loaded.
