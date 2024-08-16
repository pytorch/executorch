## Summary
LLaVA is the first multi-modal LLM ExecuTorch supports. In this directory, we
- Host a model definition for LLaVA.
- Demonstrate how to export [LLavA](https://github.com/haotian-liu/LLaVA) multimodal model to a .pte file.
- Provide a C++ runner that loads the .pte file, the tokenizer and an image, then generate responses based on user prompt.

## Instructions
### Export .pte & other artifacts

Run the following command to generate `llava.pte`, `tokenizer.bin` and an image tensor (serialized in TorchScript) `image.pt`.

Prerequisite: run `install_requirements.sh` to install ExecuTorch and run `examples/models/llava/install_requirements.sh` to install dependencies.

```bash
python -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
```

Currently the whole export process takes about 6 minutes. We also provide a small test util to verify the correctness of the exported .pte file. Just run:

```bash
python -m executorch.examples.models.llava.test.test_pte llava.pte
```

If everything works correctly it should give you some meaningful response such as:



### Build C++ runner

Run the following cmake commands from `executorch/`:

```bash
# build libraries
cmake                                               \
    -DCMAKE_INSTALL_PREFIX=cmake-out                \
    -DCMAKE_BUILD_TYPE=Debug                        \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON          \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON     \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON            \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON         \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON         \
    -DEXECUTORCH_BUILD_XNNPACK=ON                   \
    -DEXECUTORCH_DO_NOT_USE_CXX11_ABI=ON            \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON        \
    -Bcmake-out .


cmake --build cmake-out -j9 --target install --config Debug

# build llava runner

dir=examples/models/llava
python_lib=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

cmake                                       \
    -DCMAKE_INSTALL_PREFIX=cmake-out        \
    -DCMAKE_BUILD_TYPE=Debug                \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON    \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON           \
    -DCMAKE_PREFIX_PATH="$python_lib"       \
    -Bcmake-out/${dir}                      \
    ${dir}


cmake --build cmake-out/${dir} -j9 --config Debug
```

Or simply run `.ci/scripts/test_llava.sh`.

Then you should be able to find `llava_main` binary:

```bash
cmake-out/examples/models/llava/llava_main
```

### Run LLaVA

Run:
```bash
cmake-out/examples/models/llava/llava_main --model_path=llava.pte --tokenizer_path=tokenizer.bin --image_path=image.pt --prompt="What are the things I should be cautious about when I visit here? ASSISTANT:" --seq_len=768 --temperature=0
```

You should get a response like:

```
When visiting a place like this, ...
```
