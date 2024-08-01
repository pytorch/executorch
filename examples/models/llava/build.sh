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
    -Bcmake-out .


cmake --build cmake-out -j9 --target install --config Debug

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

# cmake-out/examples/models/llava/multimodal_main                                     \
#     --tokenizer_path /data/users/larryliu/llava/tokenizer.bin                       \
#     --model_path /data/users/larryliu/llava/llava_combined_xnnpack.pte              \
#     --prompt "\nWhat are the things I should be cautious about when I visit here?"  \
#     --image_path /data/users/larryliu/llava/image.pt                                \
#     --temperature 0
