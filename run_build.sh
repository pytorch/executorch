

bash install_executorch.sh --clean

bash install_executorch.sh

pip install -e .

cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_XNNPACK=OFF \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -DEXECUTORCH_BUILD_KERNELS_TORCHAO=ON \
        -DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
        -Bcmake-out .

cmake --build cmake-out -j16 --target install --config Release

